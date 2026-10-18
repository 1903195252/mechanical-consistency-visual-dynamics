"""Editable draw.io figure with real recorded renders and native vector annotations."""
import base64
import io
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import hashlib
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon
import matplotlib.patheffects as pe
from prepare_artifacts import ROOT, PAPER
from prepare_rendered_figures import ASSETS, export

SOURCE = PAPER / 'figures/audit_mechanism.drawio'
WIDTH, HEIGHT = 1200, 500


def create_source():
    doc = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17')
    diagram = ET.SubElement(doc, 'diagram', name='Rendered physical audit', id='physical-audit')
    model = ET.SubElement(diagram, 'mxGraphModel', dx=str(WIDTH), dy=str(HEIGHT), grid='1', gridSize='5',
                          page='1', pageScale='1', pageWidth=str(WIDTH), pageHeight=str(HEIGHT), math='1')
    graph = ET.SubElement(model, 'root')
    ET.SubElement(graph, 'mxCell', id='0'); ET.SubElement(graph, 'mxCell', id='1', parent='0')

    def node(key, text, x, y, w, h, size=20, bold=False, fill='none', shape='text', color='#243747', tex=None, align='center', stroke='#7D92A1', stroke_width=1.2, hatch=False, background=False, opacity=None):
        style = ('shape=%s;html=0;whiteSpace=wrap;align=%s;verticalAlign=middle;fillColor=%s;strokeColor=%s;'
                 'fontColor=%s;fontSize=%s;fontFamily=DejaVu Sans;fontStyle=%d;strokeWidth=%s;') % (
                     shape, align, fill, 'none' if shape == 'text' else stroke, color, size, int(bold), stroke_width)
        c = ET.SubElement(graph, 'mxCell', id=key, value=text, style=style, vertex='1', parent='1')
        if hatch: c.set('style', c.get('style') + 'hatch=diagonal;')
        if background: c.set('style', c.get('style') + 'background=1;')
        if opacity is not None: c.set('style', c.get('style') + f'opacity={opacity};')
        if tex: c.set('value', re.sub(r'\$(.+?)\$', lambda m: r'\('+m.group(1)+r'\)', tex))
        ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as':'geometry'})

    def picture(key, path, x, y, w, h):
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        c = ET.SubElement(graph, 'mxCell', id=key, value='', vertex='1', parent='1',
                          style='shape=image;imageAspect=1;aspect=fixed;image=data:image/png,'+encoded+';')
        c.set('asset', str(path.relative_to(PAPER)))
        ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as':'geometry'})

    def edge(key, points, color='#536D80', arrow=True, dash=False, width=2.2, source=None, target=None, halo=False, head_size=8):
        c = ET.SubElement(graph, 'mxCell', id=key, edge='1', parent='1',
                          style='strokeColor=%s;strokeWidth=%s;endArrow=%s;endFill=1;endSize=%s;dashed=%d;' % (
                              color, width, 'block' if arrow else 'none', head_size, int(dash)))
        if source: c.set('source', source)
        if target: c.set('target', target)
        if halo: c.set('style', c.get('style')+'whiteHalo=1;jumpStyle=arc;')
        g = ET.SubElement(c, 'mxGeometry', relative='1', attrib={'as':'geometry'})
        for label, point in [('sourcePoint', points[0]), ('targetPoint', points[-1])]:
            ET.SubElement(g, 'mxPoint', x=str(point[0]), y=str(point[1]), attrib={'as':label})
        if len(points)>2:
            a = ET.SubElement(g, 'Array', attrib={'as':'points'})
            for x,y in points[1:-1]: ET.SubElement(a, 'mxPoint', x=str(x), y=str(y))

    from prepare_model_overview import export as export_overview, METRICS
    overview = export_overview()
    palette = overview['plot']['criterion_colors']
    profiles = overview['profiles']
    labels = ['KeyCLD 64', 'KeyCLD 256', 'CON 64', 'CON 256', 'V-JEPA AC', 'LingBot FT']
    blue, orange, green = palette['geometry'], palette['dynamics'], palette['energy']
    muted = '#657781'
    node('a-title', '(a) From pixels to mechanics', 8, 0, 395, 32, 21, True, align='left')
    node('b-title', '(b) Native modeling pathways', 425, 0, 425, 32, 21, True, align='left')
    node('c-title', '(c) Independent SI audit', 878, 0, 312, 32, 21, True, align='left')

    # Upper distributions use exact case-level ratios; the lower contribution
    # stacks and composite shifts retain their established layout.
    layout=overview['plot']['upper_mechanics_layout']
    prefix='mechanics-upper-'
    node(prefix+'heading','After geometric reconstruction',8,34,395,17,15,True,align='left')
    node(prefix+'planar','Model / px',8,54,90,13,11.8,True,align='left',color=muted)
    node(prefix+'count-heading','Measurable',99,54,103,14,12.3,True,color=muted)
    for metric,title in [('dynamics','Effort / tol.'),('energy','Energy / tol.')]:
        spec=layout['axes'][metric]
        left,extent=spec['left'],spec['width']
        loglo,loghi=np.log10(spec['limits'])
        def mapped(value):
            return left+extent*(np.log10(value)-loglo)/(loghi-loglo)
        hue=palette[metric]
        node(prefix+metric+'-heading',title,left-3,54,extent+6,14,12.3,True,color=hue)
        edge(prefix+metric+'-axis',[(left,layout['axis_y']),(left+extent,layout['axis_y'])],
             '#8D9AA3',False,False,1)
        threshold=mapped(1)
        edge(prefix+metric+'-tolerance',[(threshold,70),(threshold,185)],'#53616A',False,True,1.2)
        edge(prefix+metric+'-divider',[(left,146),(left+extent,146)],'#D6DDE1',False,False,.8)
        for value in spec['ticks']:
            xx=mapped(value)
            edge(prefix+metric+f'-tick-{value}',[(xx,layout['axis_y']),(xx,layout['axis_y']+3)],
                 '#8D9AA3',False,False,.9)
            label='10⁴' if value==10000 else f'{value:g}'
            node(prefix+metric+f'-tick-label-{value}',label,xx-14,191,28,15,12.0,color=muted)
        for i,(profile,y) in enumerate(zip(profiles,layout['row_y'])):
            data=profile['reconstructed_mechanics']
            q0,q1,median,q3,q4=data['distributions'][metric]['quantiles']
            assert spec['limits'][0]<q0<=q4<spec['limits'][1]
            for j,record in enumerate(data['cases']):
                token=(record['seed']+'/'+record['case_id']).encode()
                uniform=int.from_bytes(hashlib.sha256(token).digest()[:4],'big')/(2**32-1)
                yy=y+(uniform*2-1)*layout['jitter_half_width']
                xx=mapped(record[metric+'_ratio'])
                diameter=layout['point_diameter']
                node(prefix+f'{metric}-case-{i}-{j}','',xx-diameter/2,yy-diameter/2,
                     diameter,diameter,fill=hue,shape='ellipse',stroke='none',
                     opacity=layout['point_opacity_percent'])
            edge(prefix+f'{metric}-range-{i}',[(mapped(q0),y),(mapped(q4),y)],
                 '#556570',False,False,.75)
            for end,value in [('min',q0),('max',q4)]:
                edge(prefix+f'{metric}-{end}-{i}',[(mapped(value),y-2),(mapped(value),y+2)],
                     '#556570',False,False,.8)
            height=layout['box_height']
            node(prefix+f'{metric}-iqr-{i}','',mapped(q1),y-height/2,
                 mapped(q3)-mapped(q1),height,fill='none',shape='rectangle',stroke=hue,stroke_width=1.2)
            edge(prefix+f'{metric}-median-{i}',[(mapped(median),y-height/2-1),(mapped(median),y+height/2+1)],
                 '#243747',False,False,1.4)
    for i,(profile,label,y) in enumerate(zip(profiles,labels,layout['row_y'])):
        node(prefix+'model-'+str(i),label,8,y-9,91,18,14.0,align='left')
        n=profile['reconstructed_mechanics']['measurable_cases']
        total=profile['registered_cases']
        fraction=n/total
        assert 0 <= fraction <= 1
        node(prefix+'count-'+str(i),f"{n}/{total} · {100*fraction:.1f}%",
             99,y-9,103,13,11.8)
        node(prefix+'measurable-'+str(i),'',100,y+5,101*fraction,4,
             fill='#496D83',shape='rectangle',stroke='none')
        if n < total:
            node(prefix+'unavailable-'+str(i),'',100+101*fraction,y+5,101*(1-fraction),4,
                 fill='#DAB99E',shape='rectangle',stroke='none')

    node('contribution-heading','Each colored width = criterion score / 6',
         8,205,395,20,16,True,align='left')
    legend=[('Closure',67),('Velocity',70),('Accel.',63),
            ('Effort',62),('Energy',68),('Point',60)]
    x=8
    for metric,(label,w) in zip(METRICS,legend):
        node('criterion-key-'+metric,'',x,233,9,9,fill=palette[metric],shape='rectangle',stroke='none')
        node('criterion-label-'+metric,label,x+13,227,w-13,21,14,align='left')
        x+=w
    node('shift-heading','Native → reconstructed',8,252,395,22,17,True,align='left')
    left,extent=112.,272.
    refx=left+extent*.5
    edge('score-reference-50',[(refx,278),(refx,457)],'#8D9AA3',False,True,1.25)
    for i,(profile,label,y) in enumerate(zip(profiles,labels,overview['plot']['contribution_layout']['row_y'])):
        node('shift-model-'+str(i),label,8,y-8,99,18,14.2,align='left')
        for track,field,dy in [('native','components_100',-7),
                              ('reconstructed','reconstructed_components_100',5)]:
            start=left
            for metric in METRICS:
                contribution=profile[field][metric]/6
                width=extent*contribution/100
                node(f'contribution-{track}-{i}-{metric}','',start,y+dy,width,4,
                     fill=palette[metric],shape='rectangle',stroke='none')
                start+=width
            expected=left+extent*profile[track+'_score_100']/100
            assert abs(start-expected)<1e-8
        native=left+extent*profile['native_score_100']/100
        rebuilt=left+extent*profile['reconstructed_score_100']/100
        edge('score-shift-'+str(i),[(native+5,y+2),(rebuilt-6,y+2)],'#476A5E',True,False,2.4)
        node('native-end-'+str(i),'',native-4.5,y-2.5,9,9,fill='white',shape='ellipse',stroke='#344651',stroke_width=1.8)
        node('reconstructed-end-'+str(i),'',rebuilt-5,y-3,10,10,fill=green,shape='rectangle',stroke='white',stroke_width=.8)
        node('native-score-value-'+str(i),f"{profile['native_score_100']:.1f}",
             native+8,y-15,36,16,13.8,True,color='#344651')
        node('reconstructed-score-value-'+str(i),f"{profile['reconstructed_score_100']:.1f}",
             max(rebuilt+9,refx+9),y-8,39,20,15,True,color='#006B52')
    edge('score-axis',[(left,457),(left+extent,457)],'#9AA7AF',False,False,1)
    for value in (0,50,100):
        x=left+extent*value/100
        edge('score-axis-tick-'+str(value),[(x,457),(x,460)],'#9AA7AF',False,False,1)
        if value!=50:
            node('score-axis-label-'+str(value),str(value),x-14,461,28,17,13.5,color=muted)
    node('score-reference-label','50\nmax. from kinematics',
         refx-93,459,186,38,12.5,color=muted)

    # Soft channel backgrounds distinguish protocols while keeping paths visible.
    node('planar-background','',420,38,429,284,fill='#F4F7F9',shape='rectangle',stroke='none',background=True)
    node('spatial-background','',420,330,429,168,fill='#F4F7F9',shape='rectangle',stroke='none',background=True)
    node('prefix-label','Planar · 64 / 256 px · shared 3-frame prefix',433,41,400,23,15,True,align='left')
    for i in range(3): picture('rgb-'+str(i),ASSETS/('observed_%d.png'%i),438+56*i,71,45,40)
    node('given','Public geometry\n+ command rule',620,71,213,40,16,fill='white',shape='rectangle')
    node('encoder','KeyCLD\nkeypoint encoder',438,139,155,40,16,fill='#EEF4F8',shape='rectangle')
    node('keycld','Constrained ODE\nmass, potential, input',620,139,213,40,16,fill='#EEF4F8',shape='rectangle')
    node('key-reader','Permutation readout',620,191,213,21,16,color=blue,fill='#EEF4F8',shape='rectangle',stroke=blue)
    node('con-encoder','CON\nCNN β-VAE (8D)',438,243,155,42,16,fill='#FFF1E8',shape='rectangle')
    node('con-ode','Latent oscillators\n14D command input',620,243,213,42,16,fill='#FFF1E8',shape='rectangle')
    node('con-reader','Affine readout',620,297,213,21,16,color=orange,fill='#FFF1E8',shape='rectangle',stroke=orange)
    edge('prefix-key-encoder',[(516.5,111),(516.5,139)],source='rgb-1',target='encoder')
    edge('prefix-con-encoder',[(438,91),(426,91),(426,264),(438,264)],source='rgb-0',target='con-encoder')
    edge('geometry-key-input',[(821,111),(821,139)],dash=True,source='given',target='keycld')
    edge('geometry-con-input',[(620,91),(608,91),(608,218),(821,218),(821,243)],dash=True,source='given',target='con-ode')
    node('key-input-label','Φ(x), τA(ω̂)',620,112,180,26,16,color=blue,fill='#F4F7F9',tex=r'$\Phi(x),\ \tau_A(\hat\omega)$')
    node('con-input-label','14D normalized vector',620,220,180,21,15.5,color=orange,fill='#F4F7F9')
    edge('key-encoded-dynamics',[(593,159),(620,159)],color=blue,source='encoder',target='keycld',halo=True)
    edge('con-encoded-dynamics',[(593,264),(620,264)],color=orange,source='con-encoder',target='con-ode')
    edge('key-ode-reader',[(726.5,179),(726.5,191)],color=blue,source='keycld',target='key-reader')
    edge('con-ode-reader',[(726.5,285),(726.5,297)],color=orange,source='con-ode',target='con-reader')
    # All coordinate readouts join one bus before the shared SI audit.
    node('readout-merge','',858,198.5,6,6,fill='#536D80',shape='ellipse',stroke='#536D80')
    edge('key-physical-readout',[(833,201.5),(858,201.5)],color=blue,source='key-reader',target='readout-merge')
    node('con-readout-join','',858,304.5,6,6,fill='#536D80',shape='ellipse',stroke='#536D80')
    edge('con-physical-readout',[(833,307.5),(858,307.5)],color=orange,source='con-reader',target='con-readout-join')
    edge('con-join-audit',[(861,307.5),(861,201.5)],arrow=False,source='con-readout-join',target='readout-merge')
    edge('prediction-audit',[(861,201.5),(861,74),(880,74)],source='readout-merge',target='native-state')

    node('spatial-input','Spatial · Stewart / Delta · RGB + commands',433,332,400,23,15,True,align='left')
    # Box each input and intermediate representation on its computation path.
    # Shared geometry and muted borders match the planar pathway above.
    input_centers = {}
    for stem,frames,resolution,y in [('spatial','3 RGB','384 px',353),
                                     ('lingbot','Last RGB','256 px',426)]:
        # Resolution describes the RGB input; commands remain a separate input.
        # The arrow bisects the collector between the two input stems.
        node(stem+'-prefix','',434,y,285,24)
        centers=[]
        for suffix,label,x,width in [('frames',frames+' · '+resolution,434,131),
                                     ('command','Actuator commands',569,150)]:
            node(stem+'-prefix-'+suffix,label,x,y,width,20,13.2,
                 fill='white',shape='rectangle')
            center=x+width/2
            centers.append(center)
            edge(stem+'-prefix-'+suffix+'-join',[(center,y+20),(center,y+24)],
                 arrow=False,width=1.2,source=stem+'-prefix-'+suffix,target=stem+'-prefix')
        input_centers[stem]=sum(centers)/2
        edge(stem+'-prefix-collector',[(centers[0],y+24),(centers[1],y+24)],arrow=False,width=1.2)
    for stem,name,detail,y,fill in [
        ('spatial','V-JEPA 2.1 + AC','Frozen ViT · 24L AC',387,'#EEF4F8'),
        ('lingbot','LingBot-VA · 5.1B','Zero-shot / adapted',460,'#FFF1E8')]:
        node(stem+'-model','',438,y,167,36,fill=fill,shape='rectangle')
        node(stem+'-model-name',name,442,y+1,159,17,16)
        node(stem+'-model-detail',detail,442,y+19,159,16,16)
    node('spatial-features','Predicted\nfeatures',620,387,80,36,13.3,fill='white',shape='rectangle')
    node('lingbot-video','Generated\nRGB',620,460,80,36,13.3,fill='white',shape='rectangle')
    node('spatial-readout','Calibrated\nphysical readout',714,387,119,36,13.8,color=blue,fill='#EEF4F8',shape='rectangle',stroke=blue)
    node('lingbot-observer','Calibrated RGB\nphysical readout',714,460,119,36,13.8,color=orange,fill='#FFF1E8',shape='rectangle',stroke=orange)
    for stem,y,color,model_name,representation,reader,input_node,input_bottom,upstream in [
        ('spatial',405,blue,'spatial-model','spatial-features','spatial-readout','spatial-prefix',377,'con-readout-join'),
        ('lingbot',478,orange,'lingbot-model','lingbot-video','lingbot-observer','lingbot-prefix',450,'spatial-readout-join')]:
        edge(stem+'-input-model',[(input_centers[stem],input_bottom),(input_centers[stem],y-18)],
             width=1.6,head_size=6,source=input_node,target=model_name)
        edge(stem+'-representation',[(605,y),(620,y)],color=color,source=model_name,target=representation)
        edge(stem+'-representation-readout',[(700,y),(714,y)],color=color,source=representation,target=reader)
        node(stem+'-readout-join','',858,y-3,6,6,fill='#536D80',shape='ellipse',stroke='#536D80')
        edge(stem+'-readout-audit',[(833,y),(858,y)],color=color,source=reader,target=stem+'-readout-join')
        upstream_y=307.5 if stem=='spatial' else 405
        edge(stem+'-join-audit',[(861,y),(861,upstream_y)],arrow=False,
             source=stem+'-readout-join',target=upstream)

    # One audit graph serves native and separately reconstructed trajectories.
    node('native-state','',880,42,166,72,fill='#F5F8FA',shape='rectangle',stroke='#7D92A1')
    node('state-title','SI state readout',880,44,166,20,15.2,True)
    node('state-values','x̂, v̂, â',880,68,166,28,18,tex=r'$\hat x,\hat v,\hat a$')
    node('state-tracks','Native / reconstructed',880,98,166,16,12.8,color=muted)
    node('si-parameters','',1064,42,125,72,fill='#F1ECF7',shape='rectangle',stroke='#8D74A7',stroke_width=1.5)
    node('si-parameters-title','Evaluator-only',1064,44,125,19,14.0,True,color='#73558D')
    node('si-parameters-values','m_i, I_i, b_i, f_ci',1064,64,125,26,15.5,
         tex=r'$m_i,I_i,b_i,f_{c_i}$')
    node('si-reference-truth','x*(t)',1064,89,125,24,14.5,tex=r'$x^*(t)$')

    node('kinematic-audit','',900,128,265,106,fill='white',shape='rectangle',stroke=blue)
    node('geometric-test','Check kinematics',912,130,241,22,15.8,True,color=blue,align='left')
    node('constraints','Φ(x̂), Φ̇(x̂), Φ̈(x̂)',912,153,156,28,15.5,color=blue,
         tex=r'$\Phi(\hat x),\quad\dot\Phi(\hat x),\quad\ddot\Phi(\hat x)$')
    node('kinematic-metric-divider','',1074,155,.6,24,fill='#CBD6DD',shape='rectangle',stroke='none')
    node('point-label','Point RMSE\nvs. reference',1079,151,79,32,12.0,color=palette['prediction'])
    # A booktabs-style coverage summary separates observed eligibility from the
    # two measurement branches above, without adding another enclosing box.
    node('coverage-rule','',912,184,241,.7,fill='#AFC4D0',shape='rectangle',stroke='none')
    node('native-coverage-label','Native mechanics coverage',912,186,241,16,12.4,True,color=muted,align='left')
    for key,label,count,x in [('planar','Planar','0/324',912),
                              ('ac','AC','0/900',994),
                              ('lingbot','LingBot','0/900',1076)]:
        node('native-coverage-family-'+key,label,x,203,77,13,11.8,color=muted)
        node('native-coverage-count-'+key,count,x,216,77,17,14.0,True)
    edge('state-kinematics',[(963,114),(963,128)],source='native-state',target='kinematic-audit')
    # The reference bus supplies truth for point error and independent SI mechanics.
    node('si-reference-join','',1179.5,180.5,5,5,fill='#8D74A7',shape='ellipse',stroke='#8D74A7')
    edge('reference-join',[(1126.5,114),(1126.5,123),(1182,123),(1182,183)],
         color='#8D74A7',arrow=False,dash=True,source='si-parameters',target='si-reference-join')
    edge('reference-kinematics',[(1182,183),(1165,183)],color='#8D74A7',dash=True,
         source='si-reference-join',target='kinematic-audit')

    node('si-mechanics','',900,252,265,82,fill='white',shape='rectangle',stroke='#718895')
    node('mechanics-operation','Map states; assemble M, c, G, Q',900,254,265,24,14.1,True,
         tex=r'Map states; assemble $M,c,G,Q$')
    node('coordinate-map','x̂, v̂, â → q̂, q̂̇, q̂̈',900,278,265,32,16.3,
         tex=r'$\hat x,\hat v,\hat a\ \rightarrow\ \hat q,\widehat{\dot q},\widehat{\ddot q}$')
    node('generalized-force','Q = Bu_A + Qext + Qf',900,312,265,22,16.3,
         tex=r'$Q=B u_A+Q_{\rm ext}+Q_f$')
    edge('kinematics-mechanics',[(1032.5,234),(1032.5,252)],source='kinematic-audit',target='si-mechanics')
    node('mechanical-eligibility','if measurable',1043,234,122,18,12.5,color=muted)
    edge('reference-mechanics',[(1182,183),(1182,287),(1165,287)],color='#8D74A7',dash=True,
         source='si-reference-join',target='si-mechanics')

    node('balance-fork','',1030,335.5,5,5,fill='#536D80',shape='ellipse',stroke='#536D80')
    edge('mechanics-balances',[(1032.5,334),(1032.5,338)],arrow=False,source='si-mechanics',target='balance-fork')
    node('effort-audit','',900,350,124,74,fill='white',shape='rectangle',stroke=orange)
    node('force-label','Project effort',900,352,124,20,14.1,True,color=orange)
    node('force-projection','Jγ = 0, JN = 0',900,375,124,21,14.5,
         tex=r'$J\gamma=0,\ JN=0$')
    node('force-equation','rτ, ru',900,400,124,24,17.0,color=orange,tex=r'$r_\tau,\ r_u$')
    node('energy-audit','',1042,350,123,74,fill='white',shape='rectangle',stroke=green)
    node('energy-label','Integrate work',1042,352,123,20,13.9,True,color=green)
    node('energy-work','W = ∫Qᵀq̇dt',1042,373,123,26,14.0,
         tex=r'$W=\int Q^{\mathsf{T}}\widehat{\dot q}\,dt$')
    node('energy-equation','rE = ΔE − W',1042,402,123,22,14.5,color=green,
         tex=r'$r_E=\Delta E-W$')
    edge('mechanics-effort',[(1032.5,338),(962,338),(962,350)],color=orange,
         source='balance-fork',target='effort-audit')
    edge('mechanics-energy',[(1032.5,338),(1103.5,338),(1103.5,350)],color=green,
         source='balance-fork',target='energy-audit')

    node('audit-report','',900,446,265,52,fill='#F5F8FA',shape='rectangle',stroke='#718895')
    node('report-criteria','Normalize six SI errors',900,449,265,21,15.0,True)
    node('report-score','Scomp · coverage · S∧',900,472,265,24,16.2,
         tex=r'$S_{\rm comp}$ · coverage · $S_\wedge$')
    node('balance-merge','',1030,427.5,5,5,fill='#536D80',shape='ellipse',stroke='#536D80')
    edge('effort-report',[(962,424),(962,430),(1032.5,430)],color=orange,arrow=False,
         source='effort-audit',target='balance-merge')
    edge('energy-report',[(1103.5,424),(1103.5,430),(1032.5,430)],color=green,arrow=False,
         source='energy-audit',target='balance-merge')
    edge('balances-report',[(1032.5,430),(1032.5,446)],source='balance-merge',target='audit-report')
    edge('kinematics-report',[(900,159),(886,159),(886,472),(900,472)],color=blue,
         source='kinematic-audit',target='audit-report')

    # Keep every connector attached to its semantic nodes in the editable source.
    cells = {cell.get('id'): cell for cell in graph}
    for cell in graph:
        if cell.get('edge') != '1': continue
        for side, point_name, prefix in [('source','sourcePoint','exit'), ('target','targetPoint','entry')]:
            if cell.get(side) is None: continue
            geometry = cells[cell.get(side)].find('mxGeometry')
            point = cell.find("mxGeometry/mxPoint[@as='%s']" % point_name)
            xx = (float(point.get('x'))-float(geometry.get('x')))/float(geometry.get('width'))
            yy = (float(point.get('y'))-float(geometry.get('y')))/float(geometry.get('height'))
            cell.set('style',cell.get('style')+'%sX=%g;%sY=%g;%sPerimeter=0;'%(prefix,xx,prefix,yy,prefix))
    ET.ElementTree(doc).write(SOURCE, encoding='utf-8', xml_declaration=True)


def render():
    if not SOURCE.exists(): create_source()
    root = ET.parse(SOURCE).getroot()
    model = root.find('.//mxGraphModel')
    width,height = float(model.get('pageWidth')),float(model.get('pageHeight'))
    fig,ax = plt.subplots(figsize=(7.05,7.05*height/width))
    ax.set_xlim(0,width); ax.set_ylim(height,0); ax.axis('off'); factor=7.05*72/width
    bounded_text = []
    arrowheads = []
    for cell in root.iter('mxCell'):
        if not cell.get('style'): continue
        style = dict(s.split('=',1) for s in cell.get('style').split(';') if '=' in s)
        g = cell.find('mxGeometry')
        if cell.get('edge') == '1':
            first=g.find("mxPoint[@as='sourcePoint']"); last=g.find("mxPoint[@as='targetPoint']")
            pts=[(float(first.get('x')),float(first.get('y')))]+[(float(p.get('x')),float(p.get('y'))) for p in g.findall('Array/mxPoint')]+[(float(last.get('x')),float(last.get('y')))]
            shaft=np.asarray(pts,dtype=float)
            color=style.get('strokeColor','#536D80'); lw=float(style.get('strokeWidth',2))*factor
            if style.get('endArrow')!='none':
                # Explicit heads retain their proportions on short connections.
                head_length=float(style.get('endSize',8))
                direction=shaft[-1]-shaft[-2];length=np.linalg.norm(direction)
                assert length>=head_length+4,('Arrow needs a visible shaft',cell.get('id'),length)
                direction/=length;normal=np.array([-direction[1],direction[0]])
                tip=shaft[-1].copy();base=tip-head_length*direction
                shaft[-1]=base
                head=Polygon([tip,base+head_length*.5*normal,base-head_length*.5*normal],
                             closed=True,facecolor=color,edgecolor='none',zorder=3)
                ax.add_patch(head)
                arrowheads.append((cell.get('id'),head))
            xs,ys=shaft.T
            halo = style.get('whiteHalo') == '1'
            artists = ax.plot(xs,ys,color=color,lw=lw,ls=(0,(4,3)) if style.get('dashed')=='1' else '-',solid_capstyle='round',zorder=2 if halo else 1)
            if halo: artists[0].set_path_effects([pe.Stroke(linewidth=lw+2,foreground='white'),pe.Normal()])
            continue
        x,y,w,h=[float(g.get(k)) for k in ('x','y','width','height')]
        shape=style.get('shape'); fill=style.get('fillColor','none'); line=style.get('strokeColor','none')
        if shape=='image':
            # Read the image embedded in the editable diagram, so image edits persist.
            encoded=style['image'].split(',',1)[1]
            im=Image.open(io.BytesIO(base64.b64decode(encoded)))
            ax.imshow(im,extent=(x,x+w,y+h,y),aspect='auto',interpolation='none',zorder=0)
            continue
        if shape=='rectangle':
            patch=Rectangle((x,y),w,h,facecolor=fill,edgecolor=line,
                            hatch='////' if style.get('hatch') else None,
                            lw=float(style.get('strokeWidth',1.2))*factor,
                            zorder=-2 if style.get('background')=='1' else 2)
            ax.add_patch(patch)
        elif shape=='ellipse':ax.add_patch(Ellipse((x+w/2,y+h/2),w,h,facecolor=fill,edgecolor=line,
                lw=float(style.get('strokeWidth',1.2))*factor,
                alpha=float(style.get('opacity',100))/100,zorder=1 if 'opacity' in style else 3))
        elif shape=='text' and fill!='none':ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor='none',alpha=.92,zorder=3))
        text=re.sub(r'\\\((.*?)\\\)', lambda m: '$'+m.group(1)+'$', cell.get('value',''))
        if text:
            ha=style.get('align','center')
            artist=ax.text(x if ha=='left' else x+w/2,y+h/2,text,ha=ha,va='center',fontsize=float(style.get('fontSize',20))*factor,
                    color=style.get('fontColor','#243747'),fontweight='bold' if style.get('fontStyle')=='1' else 'normal',linespacing=1.4,zorder=4)
            bounded_text.append((cell.get('id'), artist, x,y,w,h))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflows=[]
    for key,artist,x,y,w,h in bounded_text:
        bbox=artist.get_window_extent(renderer)
        lo=ax.transData.transform((x,y));hi=ax.transData.transform((x+w,y+h))
        if bbox.width>abs(hi[0]-lo[0])+1: overflows.append((key,'width',round(bbox.width,1),round(abs(hi[0]-lo[0]),1)))
        if bbox.height>abs(hi[1]-lo[1])+1: overflows.append((key,'height',round(bbox.height,1),round(abs(hi[1]-lo[1]),1)))
    if overflows: fig.savefig(PAPER/'build/audit_mechanism_layout.png',dpi=220)
    assert not overflows,overflows
    bounds=[artist.get_window_extent(renderer).padded(.2*fig.dpi/72)
            for _,artist,*_ in bounded_text]
    overlaps=[(bounded_text[i][0],bounded_text[j][0])
              for i in range(len(bounds)) for j in range(i+1,len(bounds))
              if bounds[i].overlaps(bounds[j])]
    assert not overlaps,('Figure 1 text overlaps',overlaps)
    arrow_text_overlaps=[(key,text_key) for key,head in arrowheads
                         for text_key,artist,*_ in bounded_text
                         if head.get_window_extent(renderer).overlaps(artist.get_window_extent(renderer))]
    assert not arrow_text_overlaps,('Figure 1 arrowheads overlap text',arrow_text_overlaps)
    export(fig,'audit_mechanism',dpi=360)


if __name__=='__main__':
    import sys
    if '--rebuild-source' in sys.argv: create_source()
    render()

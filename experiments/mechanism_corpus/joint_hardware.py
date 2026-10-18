"""Explicit universal-joint solids reused by spatial mechanism families."""
import numpy as np
from .spatial_solids import part,rx,rz


def universal(prefix,base_body,cross_body,rod_body,anchor):
    result=[];anchor=np.asarray(anchor)
    def add(label,body,shape,p,d,material,R=None):result.append(part(prefix+'_'+label,body,shape,p,d,material,R))
    add('mount_foot',base_body,'box',anchor+[0,0,-.0335],{'size':[.065,.048,.027]},'frame')
    add('cross',cross_body,'box',[0,0,0],{'size':[.018,.018,.018]},'steel')
    for sign in [-1,1]:
        add('fixed_ear_'+str(sign),base_body,'bearing_block',anchor+[sign*.017,0,0],
            {'size':[.016,.010,.040],'bore_radius':.004525},'aluminum',rz(np.pi/2))
        add('mount_bolt_'+str(sign),base_body,'socket_head',anchor+[sign*.027,0,-.018],
            {'radius':.0035,'depth':.004},'steel',rx(-np.pi/2))
        for axis in ['x','y']:
            point=[sign*.0155,0,0] if axis=='x' else [0,sign*.0155,0]
            add(axis+'_pin_'+str(sign),cross_body,'cylinder',point,{'radius':.0045,'depth':.013},'steel',rz(np.pi/2) if axis=='x' else None)
            point=[sign*.0245,0,0] if axis=='x' else [0,sign*.0245,0]
            rotation=rz(sign*np.pi/2) if axis=='x' else rz(np.pi if sign==1 else 0)
            add(axis+'_cap_'+str(sign),cross_body,'socket_head',point,{'radius':.0065,'depth':.005},'steel',rotation)
        add('fork_ear_'+str(sign),rod_body,'bearing_block',[0,sign*.017,0],
            {'size':[.014,.010,.020],'bore_radius':.004525},'aluminum')
        add('fork_stem_'+str(sign),rod_body,'box',[0,sign*.017,.027],{'size':[.014,.010,.034]},'aluminum')
    add('fork_bridge',rod_body,'box',[0,0,.049],{'size':[.025,.044,.010]},'aluminum')
    add('heel_cap',rod_body,'cylinder',[0,0,.059],{'radius':.012,'depth':.010},'steel',rx(np.pi/2))
    return result

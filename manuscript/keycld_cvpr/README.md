# Manuscript

**Beyond Geometric Closure: Measuring and Diagnosing Mechanical Consistency in Visual Dynamics**

`main.tex` and `supplement.tex` contain the current manuscript. The compiled main paper has eight body pages and one reference page; the supplement has 26 pages. `figures/`, `tables/`, and `sources/` contain the figure assets, tables, and numerical inputs. CVPR bibliography formatting uses the supplied `ieeenat_fullname.bst`.

```bash
bash manuscript/keycld_cvpr/build.sh
```

Compilation uses the included figure/table assets. Regenerating experimental measurements requires the artifact bundles described in [data.md](../../docs/data.md). Plotting and consistency-check scripts are in `scripts/`.

The signed arXiv source is available at `../keycld_arxiv/source/`; its build and synchronization scripts are in that parent directory. The [repository results guide](../../docs/results.md) describes all reported cohorts and measurements.

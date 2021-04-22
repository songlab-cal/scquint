from setuptools import setup


install_requires = [
    "anndata",
    "scanpy",
    "networkx",
    "torch",
    "pyro-ppl",
    "scikit-learn",
    "snakemake",
    "statsmodels",
    "scikit-bio",
]


setup(name='scquint',
      version='0.1',
      description='scQuint',
      url='http://github.com/songlab-cal/scquint',
      author='Gonzalo Benegas',
      author_email='gbenegas@berkeley.edu',
      license='MIT',
      packages=['scquint'],
      scripts=["scripts/sjFromSAMcollapseUandM_filter_min_overhang.awk"],
      zip_safe=False,
      install_requires=install_requires)

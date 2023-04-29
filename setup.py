from setuptools import setup, find_packages

pkg_name = 'ai_clinician'

setup(name=pkg_name,
      version='0.1', 
      packages=find_packages(),
      install_requires=[
          'numpy>=1.24.0',
          'pandas>=1.5.0',
          'scikit-learn>=1.2.0',
          'scipy>=1.4.1',
          'tqdm>=4.61.2',
          'matplotlib>=3.3.4',
          'google-auth-oauthlib>=0.4.6',
          'google-cloud-bigquery>=2.30.1',
          'pyarrow>=6.0.0',
          'numba>=0.53.0',
          'torch==1.13.0',
          'signatory==1.2.3.1.6.0',
          'torchdiffeq==0.1.1',
          'h5py>=3.6.0',
          'torchvision',
      ])
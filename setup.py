from setuptools import setup, find_packages

pkg_name = 'ai_clinician'

setup(name=pkg_name,
      version='0.1', 
      packages=find_packages(),
      install_requires=[
          'numpy>=1.19.0',
          'pandas>=1.0.5',
          'sklearn>=0.0',
          'scipy>=1.4.1',
          'tqdm>=4.61.2',
          'matplotlib>=3.3.4',
          'google-auth-oauthlib>=0.4.6',
          'google-cloud-bigquery>=2.30.1',
          'pyarrow>=7.0.0'
      ])
import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

setuptools.setup(
      name='cohlib', 
      version='0.0.1', 
      description='Resources for spike field coherence', 
      long_description=long_description,
      long_description_content_type='text/markdown',
      author= 'John Tauber',
      author_email='jtauber@mit.edu',
      license='LICENSE', 
    #   packages=['nsrl/load_data','nsrl/signal_processing','nsrl/utilities'], #include all packages under nsrl
      packages=setuptools.find_packages(),
    #   package_dir={'':'.'}, #tell disutils packages are under nsrl
    #   include_package_data=True, #include everything in source control
    #   exclude_package_data={'': ['README.rst']}, #do not include README
    #   install_requires=[
    #       'numpy', 'scikit-learn', 'matplotlib', 'scipy', 'nitime', 'pandas'
    #   ],
      classifiers = [
          'Intended Audience :: Me',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS',
      ],
      zip_safe=False)
from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='FingerprintDecoder',
      version='0.1',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Islambek Ashyrmamatov',
      author_email='ashyrmamatov01@gmail.com',
      license='Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)',
      packages=['FingerprintDecoder'],
      install_requires=[
          'rdkit-pypi==2021.3.4',
          'torch==1.9.0',
          'selfies==1.0.4',
          'sentencepiece==0.1.95',
          'pandas>=1.3.1',
          'numpy>=1.22.1',
          'matplotlib>=3.4.3',
          'seaborn==0.11.1'
      ],
      entry_points = '''
            [console_scripts]
            train = FingerprintDecoder.train:main
            evaluate = FingerprintDecoder.evaluate:main
            predict = FingerprintDecoder.predict:main
      ''',
      zip_safe=False
)


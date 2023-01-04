from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='MolForge',
      version='1.0.2',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Umit V. Ucak, Islambek Ashyrmamatov, Juyong Lee',
      author_email='{braket, ashyrmamatov, nicole23}@snu.ac.kr',
      license='Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)',
      url='https://github.com/knu-lcbc/MolForge',
      packages=['MolForge'],
      install_requires=[
          'selfies>=1.0.4',
          'sentencepiece',
          'pandas>=1.3.1',
          'numpy>=1.22.1',
          'matplotlib>=3.4.3',
          'seaborn>=0.11.1',
          'gdown'
      ],
      entry_points = '''
            [console_scripts]
            train = MolForge.train:main
            evaluate = MolForge.evaluate:main
            predict = MolForge.predict:main
            download_checkpoints = MolForge.utils:download_checkpoints
      ''',
      zip_safe=True
)


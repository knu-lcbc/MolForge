from setuptools import setup

setup(name='FingerprintDecoder',
      version='0.1',
      desctiption='The package setup',
      author='Islambek Ashyrmamatov',
      author_email='ashyrmamatov01@gmail.com',
      license='Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)',
      packages=['src'],
      install_requires=[
          'rdkit-pypi==2021.03.4',
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
            train = src.train:main
            evaluate = src.evaluate:main
            predict = src.predict:main
      ''',
      zip_safe=False
)


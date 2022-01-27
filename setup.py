from setuptools import setup

setup(name='FingerprintDecoder',
      version='0.1',
      desctiption='The package setup',
      author='Islambek Ashyrmamatov',
      author_email='ashyrmamatov01@gmail.com',
      license='Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)',
      packages=['src'],
      entry_points = '''
            [console_scripts]
            train = src.train:main
            evaluate = src.evaluate:main
            predict = src.predict:main
      ''',
      zip_safe=False
)


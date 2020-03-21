import setuptools

VERSION = '0.1.2'
SHORT_DESCRIPTION = 'Light, modular framework for dynamic time series modeling'
URL = 'https://github.com/fl16180/nowcast'

setuptools.setup(
    name='nowcast',
    version=VERSION,
    author='Fred Lu',
    author_email='fredlu.flac@gmail.com',
    description=SHORT_DESCRIPTION,
    long_description='See GitHub for description',
    url=URL,
    license='GPL-3.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.17.1',
        'scipy>=1.2.1',
        'scikit-learn>=0.20.3',
        'pandas>=0.25.1',
        'tqdm>=4.36.1'
    ]
)

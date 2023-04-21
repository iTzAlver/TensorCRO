# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import setuptools

with open('./README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

REQUIREMENTS = [
    'numpy>=1.23.5',
    'tensorflow>=2.12.0',
    'matplotlib>=3.7.1',
    'pandas>=2.0.0',
]


setuptools.setup(
    name='tensorcro',
    version='1.1.5',
    author='Palomo-Alonso, Alberto',
    author_email='a.palomo@edu.uah',
    description='TensorCRO: A Tensorflow-based implementation of the Coral Reef Optimization algorithm.',
    keywords='deeplearning, ml, api, optimization, heuristic',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/iTzAlver/tensorcro.git',
    project_urls={
        'Documentation': 'https://htmlpreview.github.io/?https://github.com/iTzAlver/tensorcro/blob/'
                         'main/',
        'Bug Reports': 'https://github.com/iTzAlver/tensorcro/issues',
        'Source Code': 'https://github.com/iTzAlver/tensorcro.git',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': './src'},
    packages=setuptools.find_packages(where='./src'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License'
    ],
    python_requires='>=3.8',
    # install_requires=['Pillow'],
    extras_require={
        'dev': ['check-manifest'],
    },
    include_package_data=True,
    install_requires=REQUIREMENTS
)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #

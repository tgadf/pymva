from setuptools import setup

setup(
    name='mtbox',
    version='0.2.4',
    author='Changhyeok Lee',
    author_email='changhyeok.lee@anthem.com',
    url='unknown',
    description='Modeling toolbox',
    long_description='Modeling toolbox',
    license='Proprietary',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'License :: Other/Proprietary License',
        'Operating System :: POSIX :: Linux'
    ],
    packages=['mtbox'],
    entry_points={
          'console_scripts': ['mtbox=mtbox.mtbox:main'],
    },
    test_suite='mtbox.tests',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False
)

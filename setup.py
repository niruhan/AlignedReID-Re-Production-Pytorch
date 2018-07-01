from setuptools import setup, find_packages


# setup parameters for google cloud ml engine
setup(
    name='tester',
    version='0.1',
    packages=find_packages(),
    description='testing on market1501',
    author='niruhan',
    include_package_data=True,
    install_requires=[
        'torchvision',
        'h5py',
        'imutils',
        'numpy',
        'opencv-python',
        'scikit-learn',
        'scipy',
        'tensorboardx',
        'googleappenginecloudstorageclient'
    ],
    zip_safe=False
)
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pdor',  # Package name for pip installation
    version='0.1',  # Current version number
    install_requires=[
        'PyPDF2',
        'pdf2image',
        'opencv-python',
        'numpy',
        'simpsave',
        'pandas',
        'openpyxl',
        'pyyaml',
        'toml',
        'requests',
    ],  # Dependencies list based on imports in your code
    packages=find_packages(),  # Automatically discover packages
    author='WaterRun',  # Author name from your code comments
    author_email='2263633954@qq.com',  # Using the same email as in the reference
    description='PDF OCR Recognition tool for automatic document extraction and analysis.',  # Brief description
    long_description=long_description,  # Long description from README.md
    long_description_content_type='text/markdown',  # Long description format
    url='https://github.com/Water-Run/pdor',  # Project homepage (assumed)
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',  # Assuming MIT license like lua-to-exe
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: General',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.10',  # Python version requirement based on modern syntax in code
    include_package_data=True,  # Include other data files
    entry_points={
        'console_scripts': [
            'pdor=pdor.pdor_out:PdorOut',  # Command line entry point
        ],
    },
)

from setuptools import setup, find_packages

# Function to read requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as req_file:
        return req_file.read().splitlines()

setup(
    name='AlloyGPT',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    description='AlloyGPT: End-to-end prediction and design of additively manufacturable alloys using an autoregressive language model',
    author='Bo Ni',
    url='https://github.com/Bo-Ni/AlloyGPT_InternalTest_0',
)
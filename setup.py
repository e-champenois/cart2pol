from setuptools import setup

if __name__ == '__main__':
	setup(
		  name='cart2pol',
		  keywords='cart2pol, cartesian, polar, coordinates',
		  packages=['cart2pol'],
		  install_requires=['numpy', 'h5py', 'scipy', 'quadrant'],
		  version='1.0.0',
		  description='Image rebinning between cartesian to polar coordinates.',
		  url='https://github.com/e-champenois/cart2pol',
		  author='Elio Champenois',
		  author_email='elio.champenois@gmail.com',
		  license='MIT',
		  zip_safe=False
		  )
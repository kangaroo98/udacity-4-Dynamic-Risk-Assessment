'''
Exception Handling - Definition

Author: Oliver
Date: 2022 - MArch
'''

# define user-defined exceptions
class AppError(Exception):
    """Base class for other exceptions"""
    pass

class UnsupportedFileType(AppError):
    """Base class for other exceptions"""
    pass
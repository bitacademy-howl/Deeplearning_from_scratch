from sklearn.externals import joblib

obj = {'data1':'data1', 'data2':123, 'data3':True}

file_name = 'object_01.pkl'
joblib.dump(obj, file_name)

file_name = 'object_01.pkl'
obj = joblib.load(file_name)


#############################################################################################################
# # 실제 아래의 코드 동작
# def dump(value, filename, compress=0, protocol=None, cache_size=None):
#     """Persist an arbitrary Python object into one file.
#
#     Parameters
#     -----------
#     value: any Python object
#         The object to store to disk.
#     filename: str or pathlib.Path
#         The path of the file in which it is to be stored. The compression
#         method corresponding to one of the supported filename extensions ('.z',
#         '.gz', '.bz2', '.xz' or '.lzma') will be used automatically.
#     compress: int from 0 to 9 or bool or 2-tuple, optional
#         Optional compression level for the data. 0 or False is no compression.
#         Higher value means more compression, but also slower read and
#         write times. Using a value of 3 is often a good compromise.
#         See the notes for more details.
#         If compress is True, the compression level used is 3.
#         If compress is a 2-tuple, the first element must correspond to a string
#         between supported compressors (e.g 'zlib', 'gzip', 'bz2', 'lzma'
#         'xz'), the second element must be an integer from 0 to 9, corresponding
#         to the compression level.
#     protocol: positive int
#         Pickle protocol, see pickle.dump documentation for more details.
#     cache_size: positive int, optional
#         This option is deprecated in 0.10 and has no effect.
#
#     Returns
#     -------
#     filenames: list of strings
#         The list of file names in which the data is stored. If
#         compress is false, each array is stored in a different file.
#
#     See Also
#     --------
#     joblib.load : corresponding loader
#
#     Notes
#     -----
#     Memmapping on load cannot be used for compressed files. Thus
#     using compression can significantly slow down loading. In
#     addition, compressed files take extra extra memory during
#     dump and load.
#
#     """
#
#     if Path is not None and isinstance(filename, Path):
#         filename = str(filename)
#
#     is_filename = isinstance(filename, _basestring)
#     is_fileobj = hasattr(filename, "write")
#
#     compress_method = 'zlib'  # zlib is the default compression method.
#     if compress is True:
#         # By default, if compress is enabled, we want to be using 3 by default
#         compress_level = 3
#     elif isinstance(compress, tuple):
#         # a 2-tuple was set in compress
#         if len(compress) != 2:
#             raise ValueError(
#                 'Compress argument tuple should contain exactly 2 elements: '
#                 '(compress method, compress level), you passed {}'
#                 .format(compress))
#         compress_method, compress_level = compress
#     else:
#         compress_level = compress
#
#     if compress_level is not False and compress_level not in range(10):
#         # Raising an error if a non valid compress level is given.
#         raise ValueError(
#             'Non valid compress level given: "{}". Possible values are '
#             '{}.'.format(compress_level, list(range(10))))
#
#     if compress_method not in _COMPRESSORS:
#         # Raising an error if an unsupported compression method is given.
#         raise ValueError(
#             'Non valid compression method given: "{}". Possible values are '
#             '{}.'.format(compress_method, _COMPRESSORS))
#
#     if not is_filename and not is_fileobj:
#         # People keep inverting arguments, and the resulting error is
#         # incomprehensible
#         raise ValueError(
#             'Second argument should be a filename or a file-like object, '
#             '%s (type %s) was given.'
#             % (filename, type(filename))
#         )
#
#     if is_filename and not isinstance(compress, tuple):
#         # In case no explicit compression was requested using both compression
#         # method and level in a tuple and the filename has an explicit
#         # extension, we select the corresponding compressor.
#         if filename.endswith('.z'):
#             compress_method = 'zlib'
#         elif filename.endswith('.gz'):
#             compress_method = 'gzip'
#         elif filename.endswith('.bz2'):
#             compress_method = 'bz2'
#         elif filename.endswith('.lzma'):
#             compress_method = 'lzma'
#         elif filename.endswith('.xz'):
#             compress_method = 'xz'
#         else:
#             # no matching compression method found, we unset the variable to
#             # be sure no compression level is set afterwards.
#             compress_method = None
#
#         if compress_method in _COMPRESSORS and compress_level == 0:
#             # we choose a default compress_level of 3 in case it was not given
#             # as an argument (using compress).
#             compress_level = 3
#
#     if not PY3_OR_LATER and compress_method in ('lzma', 'xz'):
#         raise NotImplementedError("{} compression is only available for "
#                                   "python version >= 3.3. You are using "
#                                   "{}.{}".format(compress_method,
#                                                  sys.version_info[0],
#                                                  sys.version_info[1]))
#
#     if cache_size is not None:
#         # Cache size is deprecated starting from version 0.10
#         warnings.warn("Please do not set 'cache_size' in joblib.dump, "
#                       "this parameter has no effect and will be removed. "
#                       "You used 'cache_size={}'".format(cache_size),
#                       DeprecationWarning, stacklevel=2)
#
#     if compress_level != 0:
#         with _write_fileobject(filename, compress=(compress_method,
#                                                    compress_level)) as f:
#             NumpyPickler(f, protocol=protocol).dump(value)
#     elif is_filename:
#         with open(filename, 'wb') as f:
#             NumpyPickler(f, protocol=protocol).dump(value)
#     else:
#         NumpyPickler(filename, protocol=protocol).dump(value)
#
#     # If the target container is a file object, nothing is returned.
#     if is_fileobj:
#         return
#
#     # For compatibility, the list of created filenames (e.g with one element
#     # after 0.10.0) is returned by default.
#     return [filename]
#
#
#
# ###############################################################################################################
# def _unpickle(fobj, filename="", mmap_mode=None):
#     """Internal unpickling function."""
#     # We are careful to open the file handle early and keep it open to
#     # avoid race-conditions on renames.
#     # That said, if data is stored in companion files, which can be
#     # the case with the old persistence format, moving the directory
#     # will create a race when joblib tries to access the companion
#     # files.
#     unpickler = NumpyUnpickler(filename, fobj, mmap_mode=mmap_mode)
#     obj = None
#     try:
#         obj = unpickler.load()
#         if unpickler.compat_mode:
#             warnings.warn("The file '%s' has been generated with a "
#                           "joblib version less than 0.10. "
#                           "Please regenerate this pickle file."
#                           % filename,
#                           DeprecationWarning, stacklevel=3)
#     except UnicodeDecodeError as exc:
#         # More user-friendly error message
#         if PY3_OR_LATER:
#             new_exc = ValueError(
#                 'You may be trying to read with '
#                 'python 3 a joblib pickle generated with python 2. '
#                 'This feature is not supported by joblib.')
#             new_exc.__cause__ = exc
#             raise new_exc
#         # Reraise exception with Python 2
#         raise
#
#     return obj
#
#
# def load(filename, mmap_mode=None):
#     """Reconstruct a Python object from a file persisted with joblib.dump.
#
#     Parameters
#     -----------
#     filename: str or pathlib.Path
#         The path of the file from which to load the object
#     mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
#         If not None, the arrays are memory-mapped from the disk. This
#         mode has no effect for compressed files. Note that in this
#         case the reconstructed object might not longer match exactly
#         the originally pickled object.
#
#     Returns
#     -------
#     result: any Python object
#         The object stored in the file.
#
#     See Also
#     --------
#     joblib.dump : function to save an object
#
#     Notes
#     -----
#
#     This function can load numpy array files saved separately during the
#     dump. If the mmap_mode argument is given, it is passed to np.load and
#     arrays are loaded as memmaps. As a consequence, the reconstructed
#     object might not match the original pickled object. Note that if the
#     file was saved with compression, the arrays cannot be memmaped.
#     """
#     if Path is not None and isinstance(filename, Path):
#         filename = str(filename)
#
#     if hasattr(filename, "read"):
#         fobj = filename
#         filename = getattr(fobj, 'name', '')
#         with _read_fileobject(fobj, filename, mmap_mode) as fobj:
#             obj = _unpickle(fobj)
#     else:
#         with open(filename, 'rb') as f:
#             with _read_fileobject(f, filename, mmap_mode) as fobj:
#                 if isinstance(fobj, _basestring):
#                     # if the returned file object is a string, this means we
#                     # try to load a pickle file generated with an version of
#                     # Joblib so we load it with joblib compatibility function.
#                     return load_compatibility(fobj)
#
#                 obj = _unpickle(fobj, filename, mmap_mode)
#
#     return obj
# ######################################################################################################################


3. Build succeeded on MacOS 
(#720-2022) 
See fix in here: https://github.com/wouterboomsma/eigency/issues/32. Source of solution is in https://github.com/Jonathan-LeRoux/lws/issues/7#issuecomment-505667036. 

> I'm 99% sure that this was related to this issue: pandas-dev/pandas#23424. At the very least, following the advice there fixed it for me! Basically, XCode 10 made the switch from libstdc++ to libc++, and because of some crazy shit here that is dependent on how you installed distutils, as well as whether you used to have older versions of Xcode installed, you may or may not have this bug.
> 
> To fix the bug, do the following (on Mac only!):
> I had to modify setup.py:41 to add some compile and link arguments, specifically the -stdlib=libc++ argument. It should look something like this if you want to fix it on macOS:
> 
> ext_modules = [Extension("lws",
>                              sources=[lws_module_src,"lwslib/lwslib.cpp"],
>                              include_dirs=["lwslib/",np.get_include()],
>                              language="c++",
>                              extra_compile_args=["-O3", "-stdlib=libc++"],
>                              extra_link_args=["-stdlib=libc++"])],
> @Jonathan-LeRoux Perhaps this could be incorporated into the codebase, with some sort of check for OS? It appears as though a better/more comprehensive solution to the hotfix I have posted above is outlined in this PR: pandas-dev/pandas#24274

See other notes at EOF. 

2. string fix for python3
In (https://github.com/skggm/skggm/blob/develop/inverse_covariance/quic_graph_lasso.py, line 115): 

> # Cython fix for Python3
>     # http://cython.readthedocs.io/en/latest/src/tutorial/strings.html
>     quic_mode = mode
>     if sys.version_info[0] >= 3:
>         quic_mode = quic_mode.encode("utf-8")


1. dependencies in the docker file
FROM andrewosh/binder-base

USER root

## Add dependency
RUN apt-get update
#RUN conda remove libgfortran
RUN conda update libgfortran --force
RUN conda install libgcc --force
RUN apt-get install -y libblas3gf libblas-doc libblas-dev liblapack3gf liblapack-doc liblapack-dev

## Environment variable try to fix lapack issue
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libgfortran.so.3

USER main

## Install requirements for Python 2
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt


# 0. EOF  

(#722-2022) 
Source 1: https://github.com/Jonathan-LeRoux/lws/issues/7. 
> pip install lws failed on Mac OSx Mojave fatal error: 'complex' file not found #include <complex> #7
>
>  Closed
> prathmesh36 opened this issue on 20 Jan 2019 · 10 comments
> Comments
> @prathmesh36
> prathmesh36 commented on 20 Jan 2019
> pip install lws
> Collecting lws
> Using cached https://files.pythonhosted.org/packages/3a/c7/856af2e1202e7a4c5102406196aa661edb402256e7ce2334be0c0d8afa2e/lws-1.2.tar.gz
> Building wheels for collected packages: lws
> Running setup.py bdist_wheel for lws ... error
> Complete output from command /anaconda3/bin/python -u -c "import setuptools, tokenize;file='/private/var/folders/wz/ynmzx_6s1yq9dt5gxd2qvpxm0000gn/T/pip-install-cj8ou7i7/lws/setup.py';f=getattr(tokenize, 'open', open)(file);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, file, 'exec'))" bdist_wheel -d /private/var/folders/wz/ynmzx_6s1yq9dt5gxd2qvpxm0000gn/T/pip-wheel-z65xwyrs --python-tag cp36:
> running bdist_wheel
> running build
> running build_ext
> building 'lws' extension
> creating build
> creating build/temp.macosx-10.7-x86_64-3.6
> creating build/temp.macosx-10.7-x86_64-3.6/lwslib
> gcc -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/anaconda3/include -arch x86_64 -I/anaconda3/include -arch x86_64 -Ilwslib/ -I/anaconda3/lib/python3.6/site-packages/numpy/core/include -I/anaconda3/include/python3.6m -c lws.bycython.cpp -o build/temp.macosx-10.7-x86_64-3.6/lws.bycython.o -O3
> warning: include path for stdlibc++ headers not found; pass '-std=libc++' on the command line to use the libc++ standard library instead [-Wstdlibcxx-not-found]
> In file included from lws.bycython.cpp:252:
> In file included from /anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy/arrayobject.h:4:
> In file included from /anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy/ndarrayobject.h:18:
> In file included from /anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy/ndarraytypes.h:1821:
> /anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
> #warning "Using deprecated NumPy API, disable it by "
> ^
> lws.bycython.cpp:471:14: fatal error: 'complex' file not found
> #include
> ^~~~~~~~~
> 2 warnings and 1 error generated.
> error: command 'gcc' failed with exit status 1
> 
> Failed building wheel for lws
> Running setup.py clean for lws
> Failed to build lws
> Installing collected packages: lws
> Running setup.py install for lws ... error
> Complete output from command /anaconda3/bin/python -u -c "import setuptools, tokenize;file='/private/var/folders/wz/ynmzx_6s1yq9dt5gxd2qvpxm0000gn/T/pip-install-cj8ou7i7/lws/setup.py';f=getattr(tokenize, 'open', open)(file);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, file, 'exec'))" install --record /private/var/folders/wz/ynmzx_6s1yq9dt5gxd2qvpxm0000gn/T/pip-record-1426losf/install-record.txt --single-version-externally-managed --compile:
> running install
> running build
> running build_ext
> building 'lws' extension
> creating build
> creating build/temp.macosx-10.7-x86_64-3.6
> creating build/temp.macosx-10.7-x86_64-3.6/lwslib
> gcc -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/anaconda3/include -arch x86_64 -I/anaconda3/include -arch x86_64 -Ilwslib/ -I/anaconda3/lib/python3.6/site-packages/numpy/core/include -I/anaconda3/include/python3.6m -c lws.bycython.cpp -o build/temp.macosx-10.7-x86_64-3.6/lws.bycython.o -O3
> warning: include path for stdlibc++ headers not found; pass '-std=libc++' on the command line to use the libc++ standard library instead [-Wstdlibcxx-not-found]
> In file included from lws.bycython.cpp:252:
> In file included from /anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy/arrayobject.h:4:
> In file included from /anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy/ndarrayobject.h:18:
> In file included from /anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy/ndarraytypes.h:1821:
> /anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
> #warning "Using deprecated NumPy API, disable it by "
> ^
> lws.bycython.cpp:471:14: fatal error: 'complex' file not found
> #include
> ^~~~~~~~~
> 2 warnings and 1 error generated.
> error: command 'gcc' failed with exit status 1
> 
> ----------------------------------------
> Command "/anaconda3/bin/python -u -c "import setuptools, tokenize;file='/private/var/folders/wz/ynmzx_6s1yq9dt5gxd2qvpxm0000gn/T/pip-install-cj8ou7i7/lws/setup.py';f=getattr(tokenize, 'open', open)(file);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, file, 'exec'))" install --record /private/var/folders/wz/ynmzx_6s1yq9dt5gxd2qvpxm0000gn/T/pip-record-1426losf/install-record.txt --single-version-externally-managed --compile" failed with error code 1 in /private/var/folders/wz/ynmzx_6s1yq9dt5gxd2qvpxm0000gn/T/pip-install-cj8ou7i7/lws/
> 
> @Jonathan-LeRoux
> Owner
> Jonathan-LeRoux commented on 20 Jan 2019
> According to this thread, the fix is to install the latest updates for Xcode. Let me know if that works.
> 
> @prathmesh36
> Author
> prathmesh36 commented on 20 Jan 2019
> Thank you, it worked after installing the XCode Command line tool Nov 2, 2018 update available on this link - https://developer.apple.com/download/more/
> 
> @prathmesh36 prathmesh36 closed this as completed on 20 Jan 2019
> @TheButlah
> TheButlah commented on 26 Jun 2019
> This is failing for me too. I have tried with all the following versions:
> 
> 11.0.0.0.1.1560537986 (Xcode 11 Beta 2)
> 10.2.1.0.1.1554506761 (macOS 10.14, Xcode 10.2.1)
> 10.2.0.0.1.1552586384 (macOS 10.14, Xcode 10.2)
> 10.1.0.0.1.1539992718 (macOS 10.14, Xcode 10.1)
> I am using python version 3.7, with anaconda. The error message is as follows:
> 
> gcc -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/Users/ryan.butler/.anaconda3/envs/Proteus/include -arch x86_64 -I/Users/ryan.butler/.anaconda3/envs/Proteus/include -arch x86_64 -Ilwslib/ -I/Users/ryan.butler/.anaconda3/envs/Proteus/lib/python3.7/site-packages/numpy/core/include -I/Users/ryan.butler/.anaconda3/envs/Proteus/include/python3.7m -c lws.cpp -o build/temp.macosx-10.7-x86_64-3.7/lws.o -O3
> warning: include path for stdlibc++ headers not found; pass '-stdlib=libc++' on the command line to use the libc++ standard library instead [-Wstdlibcxx-not-found]
> In file included from lws.cpp:608:
> In file included from /Users/ryan.butler/.anaconda3/envs/Proteus/lib/python3.7/site-packages/numpy/core/include/numpy/arrayobject.h:4:
> In file included from /Users/ryan.butler/.anaconda3/envs/Proteus/lib/python3.7/site-packages/numpy/core/include/numpy/ndarrayobject.h:12:
> In file included from /Users/ryan.butler/.anaconda3/envs/Proteus/lib/python3.7/site-packages/numpy/core/include/numpy/ndarraytypes.h:1824:
> /Users/ryan.butler/.anaconda3/envs/Proteus/lib/python3.7/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:17:2: warning: "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
> #warning "Using deprecated NumPy API, disable it with " \
>  ^
> lws.cpp:828:14: fatal error: 'complex' file not found
>     #include <complex>
>              ^~~~~~~~~
> 2 warnings and 1 error generated.
> error: command 'gcc' failed with exit status 1
> make: *** [build] Error 1
> @TheButlah
> TheButlah commented on 26 Jun 2019 • 
> I'm 99% sure that this was related to this issue: pandas-dev/pandas#23424. At the very least, following the advice there fixed it for me! Basically, XCode 10 made the switch from libstdc++ to libc++, and because of some crazy shit here that is dependent on how you installed distutils, as well as whether you used to have older versions of Xcode installed, you may or may not have this bug.
> 
> To fix the bug, do the following (on Mac only!):
> I had to modify setup.py:41 to add some compile and link arguments, specifically the -stdlib=libc++ argument. It should look something like this if you want to fix it on macOS:
> 
> ext_modules = [Extension("lws",
>                              sources=[lws_module_src,"lwslib/lwslib.cpp"],
>                              include_dirs=["lwslib/",np.get_include()],
>                              language="c++",
>                              extra_compile_args=["-O3", "-stdlib=libc++"],
>                              extra_link_args=["-stdlib=libc++"])],
> @Jonathan-LeRoux Perhaps this could be incorporated into the codebase, with some sort of check for OS? It appears as though a better/more comprehensive solution to the hotfix I have posted above is outlined in this PR: pandas-dev/pandas#24274
> 
> @TheButlah TheButlah mentioned this issue on 26 Jun 2019
> Fixing broken dependencies (lws, bandmat) on macOS 10.14 r9y9/wavenet_vocoder#159
>  Closed
> @Jonathan-LeRoux
> Owner
> Jonathan-LeRoux commented on 5 Jul 2019
> Thanks for the detailed analysis and the proposed fix.
> I'm happy to incorporate some fix in the codebase. It seems to me that the strategy in the pandas PR pandas-dev/pandas#24274 is less likely to lead to errors on other platforms.
> Did you confirm it worked on your system? I won't have a Mac available to try this out for several weeks.
> 
> @Jonathan-LeRoux Jonathan-LeRoux reopened this on 5 Jul 2019
> @Jonathan-LeRoux
> Owner
> Jonathan-LeRoux commented on 5 Jul 2019
> I pushed an updated version to Github, but I'm having issues with posting to PyPI.
> twine is complaining that my README is not formatted properly, but for some reason its checking tool is converting all text to lowercase, which breaks some links that are fine otherwise...
> 
> @Jonathan-LeRoux
> Owner
> Jonathan-LeRoux commented on 5 Jul 2019
> I think there is a bug in twine when checking rst files, so I moved the README to markdown.
> 1.2.1 is now up on PyPI, please check.
> 
> @Jonathan-LeRoux Jonathan-LeRoux closed this as completed on 5 Jul 2019
> @TheButlah
> TheButlah commented on 5 Jul 2019
> OK, Ill check on this once I get access again to my macbook
> 
> @TheButlah
> TheButlah commented on 8 Jul 2019
> Yep it works! Thanks for the rapid response :)
> 
> @Jonathan-LeRoux Jonathan-LeRoux mentioned this issue on 9 Jul 2019
> FileNotFoundError (_version.py) while installing library #8
>  Closed
> @Jonathan-LeRoux Jonathan-LeRoux mentioned this issue on 9 Jul 2019
> ModuleNotFoundError: No module named '_version' when importing lws #9
>  Closed
> @matsutakk
> matsutakk commented on 17 Nov 2019 • 
> @TheButlah Your solution helped me, thank you very much.
> Issue

Source 2: https://developer.apple.com/forums/thread/100377. 
> Is it possible to do complex arithmetic in Xcode?
> I am attempting to do a neural network code that requires the capability complex analysis . The program worrks in both c and c++ but fails to compile in xcode. Apparently <complex.h> is not available. In addition, I would actually run the code using Matlab but it is my undersanding that Apple does not provide a c or c++ compiler that can be used by Matlab.
> 
> Objective-C
> Up vote post of jbarbieri1938
> Down vote post of jbarbieri1938
>  770views
> Posted 4 years ago by  jbarbieri1938
> Copy jbarbieri1938 question
> Reply
> Add a Comment
> Replies
> >Apparently <complex.h> is not available.
> 
> 
> 
> Show your 'import' related line of code...
> 
> Posted 4 years ago by  KMT
> Copy KMT answer
> Up vote reply of KMT
> Down vote reply of KMT
> Add a Comment
> Complex.h is on my machine (MacOS 10.13.4, Xcode 9.3) in /Applications/Xcode.app/... (use find to look for all the usr/include directories). That should be automatically included in Xcode search paths as part of the system search paths. Note that it is NOT in /usr/include.
> 
> 
> 
> Just build a simple command line project in Xcode that included <complex.h>, and had no problem building the project. Compiled same source code using the clang command line compiler and had no problem.
> 
> 
> 
> Also, I've been writing C++ code using the <complex> methodology (C++ operators for complex numbers, etc.) for a good long while, ever since clang started supporting C++11.
> 
> 
> 
> What version of Xcode do you have?
> 
> Posted 4 years ago by  jonprescott
> Copy jonprescott answer
> Up vote reply of jonprescott
> Down vote reply of jonprescott
> Add a Comment
> There are quite a few folks that have been using Matlab with Xcode/clang on a Mac. Setup is a little tricky because of the Xcode architecture, but, it has been done. Searching the web, especially at mathworks.com, should give you some hints. If you've installed the command line tools, it may be a little easier since clang/clang++ are installed in /usr.
> 
> 
> 
> The gcc installed is llvm-gcc 4.2.1, which is probably not suitable (installed for older code backward compatibility). If you need to use an up-to-date gcc, you'll have to build it from scratch. Homebrew is your friend here.



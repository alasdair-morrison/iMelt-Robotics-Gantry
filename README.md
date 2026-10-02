//=============================================================================  
// Copyright (c) 2026 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.  
//  
// This software is the confidential and proprietary information of FLIR  
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You  
// shall not disclose such Confidential Information and shall use it only in  
// accordance with the terms of the license agreement you entered into  
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).  
//  
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE  
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE  
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR  
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES  
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING  
// THIS SOFTWARE OR ITS DERIVATIVES.  
//  =============================================================================

# README

# TABLE OF CONTENTS

- [1. INSTALLATION](#1-installation)
  - [1.1 WINDOWS](#11-windows)
  - [1.2 LINUX](#12-linux)
  - [1.3 MACOS](#13-macos)
- [2. API DIFFERENCES](#2-api-differences)
- [3. REMOVE PYSPIN](#3-remove-pyspin)
- [4. TROUBLESHOOT](#4-troubleshoot)
  - [4.1 NumPy 2.x: Module Compatibility Notice](#41-numpy-2x-module-compatibility-notice)
  - [4.2 Matplotlib: Module Compatibility Notice](#42-matplotlib-module-compatibility-notice)
  - [4.3 Issue with NumPy 1.19.5 on Linux ARM64 when Importing PySpin](#43-issue-with-numpy-1195-on-linux-arm64-when-importing-pyspin)


PySpin is a wrapper for Spinnaker library.

Teledyne Machine Vision's website is located at https://www.teledynevisionsolutions.com/solutions/machine-vision/

The `PySpin` Python extension provides a common software interface to control and acquire images from `Teledyne USB 3.0, GigE`, and USB 2.0 cameras using the same API under 64-bit Windows.

=============================================================================
## 1. INSTALLATION
=============================================================================

-----------------------------------------------------------------------------
### 1.1 WINDOWS
-----------------------------------------------------------------------------

**Recommended: Use a Virtual Environment**

Using a virtual environment helps isolate your PySpin installation from other Python projects and 
avoids potential dependency conflicts. This is the **recommended approach** for installing PySpin.

1. Install Python
   
   Supported Python versions: `3.10`, `3.12`
   
   Download from: https://www.python.org/downloads/
   
   **Note:** The Python website defaults to 32-bit installers. For 64-bit versions,
   click into the specific release version page.

2. (Optional) Configure PATH environment variable
   
   This may be done automatically during installation. To configure manually:
   
   - Navigate to: `My Computer > Properties > Advanced System Settings > Environment Variables`
   - Add your Python installation location to the PATH variable
   
   Example: If installed at `C:\Python310\`, add:
   
       C:\Python310\

3. Create and activate a virtual environment

   Create a virtual environment for your PySpin projects:

       <python version> -m venv C:\venv\spinnaker

   Activate the virtual environment:

       C:\venv\spinnaker\Scripts\activate

   After activation, your prompt will show `(spinnaker)` prefix.
   
   **Note:** To use PySpin in the future, remember to activate this environment first:

       C:\venv\spinnaker\Scripts\activate

4. Install Spinnaker SDK prerequisites
   
   Run the `Spinnaker SDK` installer matching your PySpin version to install 
   required drivers and `Visual Studio` redistributables.
   
   Example: For `PySpin 3.0.0.0`, install `Spinnaker 3.0.0.0` first, 
   selecting only the `Visual Studio` runtimes and drivers.

5. Install dependencies within the virtual environment
   
   With your virtual environment activated, update pip:
   
       python -m ensurepip
       python -m pip install --upgrade pip
   
   **Install NumPy and Matplotlib:**
   
   `Numpy` version requirement: install NumPy 1.x (< 2.0) — NumPy 2.x is not supported.
   
   `Matplotlib` is optional and used in examples to demonstrate PySpin usage.
   
   Install command for Python < 3.12:
   
       pip install --upgrade "numpy<2" matplotlib
   
   Install command for Python 3.12+:
   
       pip install --upgrade "numpy>=2" matplotlib
   
   **Optional: Install Pillow for better image format support**
   
   Some `Pillow` versions may not support certain Python versions. See compatibility:
   https://pillow.readthedocs.io/en/stable/installation/python-support.html
   
   Example for Python 3.10:
   
       pip install Pillow==9.2.0

6. Install the PySpin wheel within the virtual environment
   
   With your virtual environment activated, install the wheel:
   
       pip install spinnaker_python-3.x.x.x-cp3x-cp3x-win_amd64.whl
   
   **Ensure the wheel file matches your Python version!**

**Running Examples:**

With your virtual environment activated, run PySpin examples directly from the command prompt:

    python Examples\Python3\Acquisition.py

**Alternative: System-wide Installation (Not Recommended)**

If you choose not to use a virtual environment, you can install PySpin system-wide:

- Update pip and install dependencies:

      <python version> -m ensurepip
      <python version> -m pip install --upgrade pip
      <python version> -m pip install --upgrade "numpy<2" matplotlib

- Install the PySpin wheel:

      <python version> -m pip install spinnaker_python-3.x.x.x-cp3x-cp3x-win_amd64.whl

- Run examples with specific Python version:

      py -3.10 Examples\Python3\Acquisition.py

-----------------------------------------------------------------------------
### 1.2 LINUX
-----------------------------------------------------------------------------

**Recommended: Use a Virtual Environment**

Modern Debian-based systems (Ubuntu, etc.) enforce PEP 668, which prevents installing packages 
into the system-wide Python environment to protect system stability. Using a virtual environment 
is the **recommended approach** for installing PySpin and its dependencies.

1. Install Python and venv module

   Ensure Python and the `venv` module are installed. On Debian/Ubuntu-based systems:

       sudo apt-get update
       sudo apt-get install python3 python3-venv

   For a specific Python version (e.g., Python 3.10):

       sudo apt-get install python3.10 python3.10-venv

2. Create and activate a virtual environment

   Create a virtual environment for your PySpin projects:

       python<version> -m venv ~/venv/spinnaker
       source ~/venv/spinnaker/bin/activate

   After activation, your prompt will show a `(spinnaker)` prefix. 
   
   **Note:** To use PySpin in the future, remember to activate this environment first:

       source ~/venv/spinnaker/bin/activate

3. Install the matching Spinnaker SDK

   Before proceeding, ensure the Spinnaker SDK Debian packages (and prerequisites) are installed.  
   Use matching versions (eg., SDK 4.0.0.0 for wheel 4.0.0.0)

4. Install dependencies within the virtual environment (NumPy, Matplotlib, optional Pillow)

   With your virtual environment activated, install the required dependencies:

   NumPy version requirements by Python version:
   - **Python < 3.12:** NumPy 1.x (< 2.0)
   - **Python 3.12+:** NumPy 2.x (>= 2.0)
   
   **Note:** The NumPy version requirement is determined by your Python version, not your operating system.
   
   `Matplotlib` is optional and used in examples to demonstrate PySpin usage.

   Install command for Python < 3.12:

       pip install --upgrade "numpy<2" matplotlib

   Install command for Python 3.12+:

       pip install --upgrade "numpy>=2" matplotlib

   **Optional: Install Pillow for better image format support**
   `Pillow` improves `Matplotlib`’s output image format support.

   Note: some versions of `Pillow` might NOT support some Python versions.  Refer to:
      https://pillow.readthedocs.io/en/stable/installation/python-support.html

   Example (Python 3.12):

       pip install Pillow==9.2.0

5. Install the PySpin wheel within the virtual environment

   With your virtual environment activated, install the PySpin wheel:

       pip install spinnaker_python-x.x.x.x-cp<pyver>-cp<pyver>-linux_x86_64.whl

   where
      <pyver> is the ABI tag used in the wheel filename (eg. 310, 312)

   Example for Python 3.10:

       pip install spinnaker_python-x.x.x.x-cp310-cp310-linux_x86_64.whl

6. Run examples

   The examples are located in the Examples folder of the extracted tarball. 
   With your virtual environment activated, run:
   
       python Examples/Python3/Acquisition.py

**Alternative: System-wide or User Installation (Not Recommended)**

If you choose not to use a virtual environment, you can install PySpin system-wide or for the current user.
However, on modern Debian-based systems, you may encounter PEP 668 protection errors.

- System-wide install (requires `sudo`):

      sudo python<version> -m pip install spinnaker_python-x.x.x.x-cp<pyver>-cp<pyver>-linux_x86_64.whl

- User-only install:

      python<version> -m pip install --user spinnaker_python-x.x.x.x-cp<pyver>-cp<pyver>-linux_x86_64.whl

- Override PEP 668 protection (not recommended, may break system Python):

      python<version> -m pip install --break-system-packages spinnaker_python-x.x.x.x-cp<pyver>-cp<pyver>-linux_x86_64.whl

You must also install dependencies using similar commands:

      python<version> -m pip install --user "numpy<2" matplotlib

where
   <version> is the Python executable version (eg. 3.10, 3.12)
   <pyver> is the ABI tag used in the wheel filename (eg. 310, 312)

-----------------------------------------------------------------------------
### 1.3 MACOS
-----------------------------------------------------------------------------

1. Check that Python is installed.  

   Supported Python versions: `3.10`, `3.12`

   There are several ways to install Up-to-date Python packages, but the recommended way is to use pyenv - the Python package manager, which manages multiple versions of Python effectively.  
   (installing Python using a method that does not use pyenv, can result in run-time errors due to mixed running Python versions)

   For example: to install the specific Python version 3.10.11 do the following steps:  
   
   - Update brew
  
         brew update

   - Install the pyenv tool
   
         brew install pyenv

   - Install the specific Python version 3.10.11

         pyenv install 3.10.11

   - Set Python version globally.

         pyenv global 3.10.11

   - Adjust the shell's path into the shell (e.g. .zshrc, .bash_profile)
    
         echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile

   - Reset the current shell
      
         source ~/.bash_profile

   - See which versions of Python are installed (e.g. * 3.10.11 (set by ~/.pyenv/version))

         pyenv versions

   - Check the Python version (e.g. Python 3.10.11)

         python3.10 -V

   - Verify that the Python uses the pyenv related path (e.g. ~/.pyenv/shims/python3.10)
   
         which python3.10


2. Update pip for Python. Run the following command for your version of Python:  

       sudo <python version> -m ensurepip

   This will install a version of pip and allow you to update or install new wheels.

3. Install library dependencies for `PySpin`: `Numpy` and `Matplotlib`.  
   
   `Numpy` version requirement: install NumPy 1.x (< 2.0) — NumPy 2.x is not supported.
   
   `Matplotlib` is not required for the library itself but is used in some of
   our examples to highlight possible usages of `PySpin`.  
   Install these dependencies by running one of the following commands.

   -  user-only install:

         <python version> -m pip install --upgrade --user numpy matplotlib

   - system-wide install (requires `sudo`):

         sudo <python version> -m pip install --upgrade numpy matplotlib

   Optional: Install `Pillow`
   `Pillow` improves `Matplotlib`’s output image format support.

   Note: some versions of `Pillow` might NOT support some Python versions.  Refer to:
      https://pillow.readthedocs.io/en/stable/installation/python-support.html

   Example (Python 3.8):

       python3.8 -m pip install Pillow==7.0.0

4. Install the matching Spinnaker SDK

   Ensure the Spinnaker SDK Debian packages (and prerequisites) are installed before installing the Python wheel.
   Use matching versions (eg., SDK 4.0.0.0 for wheel 4.0.0.0)

5. Install the PySpin wheel

   You can install the wheel system-wide (requires `sudo`).

       sudo python<version> -m pip install spinnaker_python-x.x.x.x-cp<pyver>-cp<pyver>-linux_x86_64.whl

      where
         <version> is the Python executable version (eg. 3.10, 3.12)
         <pyver> is the ABI tag used in the wheel filename (eg. 310, 312)

    Examples:
      
   - Python 3.10, system-wide:

         sudo python3.10 -m pip install spinnaker_python-x.x.x.x-cp310-cp310-linux_x86_64.whl

6. The examples are located in the Examples folder of the extracted tarball.  
   Run with: ex.
   
       python<version> Examples/Python3/Acquisition.py

=============================================================================
## 2. API DIFFERENCES
=============================================================================

Except for the changes listed below, most function names are exactly the same
as the C++ API. See examples for PySpin usage!

- All methods of SpinnakerException no longer exist, please replace all
  usages of SpinnakerException with any of the following attributes:  
    - message: Normal exception message.
    - fullmessage: Exception message including line, file, function,
                   build date, and time (from C++ library).  
    - errorcode: Integer error code of the exception.  
  
  The SpinnakerException instance itself can be printed, as it derives from
  the BaseException class and has a default __str__ representation.  
  See examples for usage.

- Image creation using NumPy arrays (although the int type of the array must be uint8)

- The majority of headers from the C++ API have been wrapped, with the exception of:
    - Headers with "Adapter" or "Port" in the name
    - NodeMapRef.h, NodeMapFactory.h
    - Synch.h, GCSynch.h, Counter.h, filestream.h

- INode and IValue types (esp. returned from GetNode()) have to
  be initialized to their respective pointer types  
  (ex. CFloatPtr, CEnumerationPtr) to access their functions

- CameraPtr, CameraList, InterfacePtr, InterfaceList, and SystemPtr  
  have to be manually released and/or deleted before program exit (use del operator)
    - See EnumerationEvents example

- Image.GetData() returns a 1-D NumPy array of integers, the int type
  depends on the pixel format of the image

- Image.GetNDArray() returns a 2 or 3-D NumPy array of integers, only for select
  image formats.  
  This can be used in libraries such as PIL and/or OpenCV.

- Node callbacks take in a callback class instead of a function pointer
    - Register is now RegisterNodeCallback, Deregister is now DeregisterNodeCallback
    - See NodeMapCallback example for more details

- IImage.CalculateChannelStatistics(StatisticsChannel channel) returns
  a ChannelStatistics object representing stats for the given channel
  in the image.  
  These stats are properties within the ChannelStatistics object.  
  Please see the docstring for details.  
  This replaces ImageStatistics!

- Pass-by-reference functions now return the type and take in void
    - GetFeatures() returns a Python list of IValue, instead of taking
      in a FeatureList_t reference
    - GetChildren() returns a Python list of INode, instead of taking
      in a NodeList_t reference
    - Same with GetEntries(), GetNodes()
    - GetPropertyNames() returns a Python list of str,
      instead of taking in a gcstring_vector reference

- Methods Get() and Set() for IRegister and register nodes use NumPy arrays
    - Get() takes in the length of the register to read and two optional
      bools, returns a NumPy array
    - Set() takes in a single NumPy array

=============================================================================
## 3. REMOVE PYSPIN
=============================================================================

Removing or updating PySpin is similar to removing or updating other wheels.

For Windows, if you need to remove PySpin, the following command needs to be
run from an administrator command prompt to remove your associated Python version:

    <python version> -m pip uninstall spinnaker-python

For Linux or MacOS, if you need to remove PySpin from a user-specific install, run
the following command to remove your associated Python version:

    <python version> -m pip uninstall spinnaker-python

For Linux or MacOS, if you need to remove PySpin from a site-wide install the
following command needs to be run as sudo to remove your associated Python version:

    sudo <python version> -m pip uninstall spinnaker-python


=============================================================================
## 4. TROUBLESHOOT
=============================================================================

-----------------------------------------------------------------------------
### 4.1 NumPy Version Compatibility Notice
-----------------------------------------------------------------------------

PySpin requires different NumPy versions depending on your **Python version**:

- **Python < 3.12:** Requires NumPy 1.x (< 2.0)
- **Python 3.12+:** Requires NumPy 2.x (>= 2.0)

This requirement is based on Python version, not on your operating system.

If you need to install or switch to the correct NumPy version:

```sh
# For Python < 3.12 (e.g., Python 3.10)
<python version> -m pip install --upgrade "numpy<2.0"

# For Python 3.12 or higher
<python version> -m pip install --upgrade "numpy>=2.0"
```

-----------------------------------------------------------------------------
### 4.2 Matplotlib: Module Compatibility Notice
-----------------------------------------------------------------------------

To ensure compatibility, you may need to upgrade related modules such as `Matplotlib` to match the 
recommended version for the installed Python version. For example:

```sh
<python version> -m pip install --upgrade pip numpy matplotlib
```

-----------------------------------------------------------------------------
### 4.3 Issue with Numpy 1.19.5 on Linux ARM64 when Importing PySpin
-----------------------------------------------------------------------------

An issue exists with `Numpy` 1.19.5 on the Linux ARM64 architecture, where importing `PySpin` can trigger an "Illegal instruction" error. This problem stems from a bug in `Numpy`, which has been resolved in version 1.20.

For further details, refer to the Numpy issue discussion.
https://github.com/numpy/numpy/issues/18131#issuecomment-794200556

#### Workarounds:

You can workaround the issue by:
- Downgrading `Numpy` to version 1.19.4
- Upgrading `Numpy` to version 1.20.x or later
- Setting the environment variable 
      ```sh
      OPENBLAS_CORETYPE=ARMV8
      ```
- Compiling from source on the failing ARM hardware

      ```sh
      pip install --no-binary :all: numpy==1.19.5
      ```

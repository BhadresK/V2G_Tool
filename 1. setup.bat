@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   V2G Simulation Tool
echo ============================================
echo.

set PYTHON_VERSION=3.11.9
set PYTHON_INSTALLER=python-installer.exe
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe

set "PYEXE="
for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYEXE (
        "%%P" --version >nul 2>nul
        if not errorlevel 1 set "PYEXE=%%P"
    )
)

if not defined PYEXE (
    echo Python not found ^(or only the Windows Store stub is present^).
    echo Installing Python %PYTHON_VERSION% for current user only...
    echo ^(No admin rights required - installs to your user profile^)
    echo.

    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing"

    if not exist "%PYTHON_INSTALLER%" (
        echo ERROR: Could not download the Python installer.
        echo This may be blocked by your company network. Please ask IT to
        echo install Python 3.10+ manually, then re-run this script.
        pause
        exit /b 1
    )

    echo Installing Python silently ^(user-level, no admin needed^)...
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0

    del "%PYTHON_INSTALLER%"

    set "PYEXE="
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
        if exist "%%D\python.exe" set "PYEXE=%%D\python.exe"
    )

    if not defined PYEXE (
        echo ERROR: Python installation could not be verified.
        echo Please install Python manually from python.org and re-run this script.
        pause
        exit /b 1
    )

    echo Python installed successfully at: !PYEXE!
) else (
    echo Found existing Python:
    "!PYEXE!" --version
)
echo.

REM ------------------------------------------------------------
REM Create virtual environment
REM ------------------------------------------------------------
if not exist ".venv" (
    echo Creating virtual environment...
    "!PYEXE!" -m venv .venv
) else (
    echo Virtual environment already exists - skipping creation.
)

set "VENV_PY=.venv\Scripts\python.exe"
echo.

REM ------------------------------------------------------------
REM Upgrade pip
REM ------------------------------------------------------------
echo Upgrading pip...
"!VENV_PY!" -m pip install --upgrade pip --no-cache-dir ^
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pythonhosted.org
echo.

REM ------------------------------------------------------------
REM Install requirements.txt
REM ------------------------------------------------------------
set MAX_ATTEMPTS=4
set ATTEMPT=1

:INSTALL_LOOP
if not exist "requirements.txt" (
    echo requirements.txt not found - skipping package install.
    goto AFTER_INSTALL
)

echo Installing packages ^(attempt !ATTEMPT!/%MAX_ATTEMPTS%^)...
"!VENV_PY!" -m pip install -r requirements.txt --no-cache-dir ^
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pythonhosted.org

if errorlevel 1 (
    if !ATTEMPT! lss %MAX_ATTEMPTS% (
        echo Install did not finish cleanly - retrying...
        set /a ATTEMPT+=1
        timeout /t 3 >nul
        goto INSTALL_LOOP
    ) else (
        echo.
        echo WARNING: Install still incomplete after %MAX_ATTEMPTS% attempts.
        echo Re-run this script again, or check your network/VPN connection.
        pause
        exit /b 1
    )
) else (
    echo All packages installed successfully.
)
:AFTER_INSTALL
echo.

REM ------------------------------------------------------------
REM Streamlit
REM ------------------------------------------------------------
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
> "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
>> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
> "%USERPROFILE%\.streamlit\config.toml" echo [browser]
>> "%USERPROFILE%\.streamlit\config.toml" echo gatherUsageStats = false
echo Streamlit pre-configured - no prompts on first run.
echo.

REM ------------------------------------------------------------
REM launcher
REM ------------------------------------------------------------
> V2G_Sim.bat echo @echo off
>> V2G_Sim.bat echo "%%~dp0.venv\Scripts\python.exe" -m streamlit run app.py
>> V2G_Sim.bat echo pause

echo ============================================
echo   Setup complete!
echo   just double-click V2G_Sim.bat
echo ============================================
pause
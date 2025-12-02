@echo off
REM Build script for Windows

set IMAGE_NAME=wan2-s2v
set VERSION=1.0.0
set DOCKER_USERNAME=%DOCKER_USERNAME%

if "%DOCKER_USERNAME%"=="" (
    echo ERROR: DOCKER_USERNAME environment variable not set
    echo Set it with: set DOCKER_USERNAME=your-dockerhub-username
    exit /b 1
)

echo ================================================
echo Building Wan2.2 S2V Docker Image
echo ================================================
echo Image: %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION%
echo.

docker build -t %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION% .
docker tag %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION% %DOCKER_USERNAME%/%IMAGE_NAME%:latest

echo.
echo Build complete!
echo.
echo Images created:
echo   - %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION%
echo   - %DOCKER_USERNAME%/%IMAGE_NAME%:latest
echo.
echo Next: Run deploy.bat to push to Docker Hub

@echo off
REM Deploy script for Windows

set IMAGE_NAME=wan2-s2v
set VERSION=1.0.0
set DOCKER_USERNAME=%DOCKER_USERNAME%

if "%DOCKER_USERNAME%"=="" (
    echo ERROR: DOCKER_USERNAME environment variable not set
    echo Set it with: set DOCKER_USERNAME=your-dockerhub-username
    exit /b 1
)

echo ================================================
echo Deploying Wan2.2 S2V to Docker Hub
echo ================================================
echo Image: %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION%
echo.

echo Pushing %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION%...
docker push %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION%

echo Pushing %DOCKER_USERNAME%/%IMAGE_NAME%:latest...
docker push %DOCKER_USERNAME%/%IMAGE_NAME%:latest

echo.
echo Deployment complete!
echo.
echo Image available at:
echo   - %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION%
echo   - %DOCKER_USERNAME%/%IMAGE_NAME%:latest
echo.
echo Next: Create RunPod template at https://www.runpod.io/console/serverless

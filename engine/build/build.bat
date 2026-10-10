@echo off
REM Builds EventCut-Engine-Setup.exe on a Windows PC.
REM Needs: Python 3.11+ (python.org, tick "Add to PATH") and Inno Setup 6 (jrsoftware.org).
REM Usage:  build\build.bat https://your-site.netlify.app
setlocal
cd /d %~dp0\..
if "%~1"=="" (echo Usage: build\build.bat https://your-site.netlify.app & exit /b 1)
echo %~1> site.txt

python -m venv .venv || exit /b 1
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt || exit /b 1

REM FFmpeg with NVIDIA encoding, subtitles and HDR tone mapping (gyan.dev full build, GPL)
if not exist bin\ffmpeg.exe (
  mkdir bin 2>nul
  powershell -NoProfile -Command "Invoke-WebRequest https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z -OutFile ffmpeg.7z" || exit /b 1
  powershell -NoProfile -Command "Invoke-WebRequest https://www.7-zip.org/a/7zr.exe -OutFile 7zr.exe" || exit /b 1
  7zr.exe e ffmpeg.7z -obin ffmpeg.exe ffprobe.exe -r -y || exit /b 1
  del ffmpeg.7z 7zr.exe
)

REM Speech-to-text model, so captions work offline from the first run
python -c "from faster_whisper import WhisperModel; WhisperModel('small', download_root='models/whisper')" 

python build\make_ico.py
pyinstaller --noconfirm --windowed --name "EventCut Engine" --icon eventcut.ico ^
  --add-data "models;models" --add-data "fonts;fonts" --add-data "bin;bin" --add-data "site.txt;." --add-data "eventcut.ico;." ^
  --collect-all faster_whisper --collect-all ctranslate2 --hidden-import pystray._win32 app.py || exit /b 1

"%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" build\eventcut.iss || exit /b 1
echo.
echo Done: build\Output\EventCut-Engine-Setup.exe
echo Upload it to your site's downloads folder.

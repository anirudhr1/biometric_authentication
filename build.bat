@echo off
echo 🔄 Cleaning old builds...

rmdir /s /q dist
rmdir /s /q build
del gui.spec 2>nul

echo 🔨 Building new .exe with PyInstaller...

python -m PyInstaller --onefile --noconsole gui.py ^
--add-data "haarcascade_frontalface_default.xml;." ^
--add-data "recognizer.yml;." ^
--add-data "labels.pkl;."

echo ✅ Build complete!
echo 💡 Your .exe is inside the 'dist' folder as 'gui.exe'
pause

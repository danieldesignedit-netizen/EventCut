; Inno Setup script for EventCut Engine
#define AppName "EventCut Engine"
#define AppVersion "1.4.2"
[Setup]
AppId={{6B7E3C1A-4E8F-4B6A-9E61-3C5D7A9B0E11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=danieldesignedit
AppPublisherURL=https://www.danieldesignedit.com
DefaultDirName={localappdata}\Programs\EventCut Engine
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename=EventCut-Engine-Setup
SetupIconFile=..\eventcut.ico
UninstallDisplayIcon={app}\EventCut Engine.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
RestartApplications=no
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\EventCut Engine\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\EventCut Engine"; Filename: "{app}\EventCut Engine.exe"
Name: "{userstartup}\EventCut Engine"; Filename: "{app}\EventCut Engine.exe"; Comment: "Starts EventCut Engine with Windows"

[Run]
Filename: "{app}\EventCut Engine.exe"; Description: "Start EventCut Engine now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "taskkill"; Parameters: "/IM ""EventCut Engine.exe"" /F"; Flags: runhidden

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#define AppName "Exo Collection System"
#define ProjectRoot AddBackslash(SourcePath) + "..\.."

[Setup]
AppId={{D4806E68-C6B3-4A3C-98CC-9054B437A1C4}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Exo Collection System Team
DefaultDirName={localappdata}\Programs\ExoCollectionSystem
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=ExoCollectionSystem-{#AppVersion}-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\ExoCollector.exe
SetupLogging=yes

[Tasks]
Name: "desktopicons"; Description: "Create desktop shortcuts"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#ProjectRoot}\dist\ExoCollector.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\dist\ExoDataStudio.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\packaging\installer\README_START_HERE.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\release\ExoCollectionSystem-{#AppVersion}-windows-x64\BUILD_MANIFEST.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\release\ExoCollectionSystem-{#AppVersion}-windows-x64\README_PROJECT.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Exo Collector"; Filename: "{app}\ExoCollector.exe"; WorkingDir: "{app}"
Name: "{group}\Exo Data Studio"; Filename: "{app}\ExoDataStudio.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Exo Collector"; Filename: "{app}\ExoCollector.exe"; WorkingDir: "{app}"; Tasks: desktopicons
Name: "{autodesktop}\Exo Data Studio"; Filename: "{app}\ExoDataStudio.exe"; WorkingDir: "{app}"; Tasks: desktopicons

[Run]
Filename: "{app}\ExoCollector.exe"; Description: "Launch Exo Collector"; Flags: nowait postinstall skipifsilent unchecked
Filename: "{app}\README_START_HERE.txt"; Description: "Open startup instructions"; Flags: postinstall shellexec skipifsilent

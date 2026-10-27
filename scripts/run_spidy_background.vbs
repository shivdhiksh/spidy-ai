' Spidy Background Launcher (VBScript)
' Runs Spidy completely hidden without a terminal/command window.

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Determine repository root directory (one level up from scripts/)
ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
RootDir = FSO.GetParentFolderName(ScriptDir)

' Check if virtual environment pythonw exists first, else use system pythonw
VenvPythonw = RootDir & "\.venv\Scripts\pythonw.exe"
If FSO.FileExists(VenvPythonw) Then
    PythonwExe = """" & VenvPythonw & """"
Else
    PythonwExe = "pythonw.exe"
End If

' Launch Spidy in background mode
Cmd = PythonwExe & " -m spidy.main --background"

' Set working directory to project root
WshShell.CurrentDirectory = RootDir

' 0 = Hide window, False = Do not wait for process to finish
WshShell.Run Cmd, 0, False

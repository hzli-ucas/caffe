setlocal enabledelayedexpansion

set path=CRNN_Adam_lr0.001
mkdir !path!

..\..\build\tools\Release\caffe.exe train --solver=solver_Adam.prototxt >!path!.log 2>&1

for /l %%i in (5,5,25) do (
move _iter_%%i0000.caffemodel !path!\
move _iter_%%i0000.solverstate !path!\
)

pause

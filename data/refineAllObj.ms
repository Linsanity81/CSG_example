--用 3ds Max Script 同时导入多个.obj文件
--1. MAXScript -> New Script, 将下列代码拷贝， 修改路径；
--2. ctl + s 保存该script文件；
--3. MAXScript -> Run Script, 选择保存的script文件， 点击Open即可。
--4. 按Ctrl+E执行


obj_files = getFiles "*.obj"

for f in obj_files do 
(
	if ( doesfileexist f ) do
		importFile f #noPrompt using:Wavefront_Object --不经提示，场景默认为前一个场景
	objfilename = getFilenameFile f
	for geo in geometry 
		where ((geo.name == objfilename) or (matchPattern geo.name pattern:"3D_Object*") or (matchPattern geo.name pattern:"default*") or (matchPattern geo.name pattern:"*_obj")) do 
	(
			--geo.name = (f as string) 
		    pathaa = filterString (f as string) "\\"
			geo.name = pathaa[pathaa.count]
	)

)


torusOBJ = getnodebyname ("Bunny_12x12x9.obj")

for geo in geometry do
(	
	if( geo != torusOBJ ) then
	(
		torusCopy = copy  torusOBJ
		ProBoolean.createBooleanObject geo torusCopy  1 3 0
		--messageBox ("Done. Objects processed: " + (objectAll as String))
	)
	else
	(
		--messageBox ("torus")
	)
	
)

delete torusOBJ


makeDir @".\\Model_Refine\\" all:false
for geo in geometry do
(
	s1 = ".\\Model_Refine\\"
	append s1 geo.name
	
	select geo 
	exportFile s1 #noPrompt selectedOnly: true
)


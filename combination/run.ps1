$env:PYTHONPATH = @(
  "D:\study\nlp\combination\.deps",
  "D:\study\nlp\9\.deps",
  "D:\study\nlp\8\.deps",
  "D:\study\nlp\7\.deps"
) -join ";"
$env:NLTK_DATA = "D:\study\nlp\combination\nltk_data"

& "C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  -m streamlit run app.py `
  --global.developmentMode=false `
  --server.port 8502 `
  --server.headless true

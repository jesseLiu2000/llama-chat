# eval

python app.py   
The model interface in app.py  
All html files are in templates folder and css / js style files are in static folder   
if you want to add new figure for in html file, please put the picture in the static folder and then follow the following format in html file:  

<img src={{ url_for('static', filename='xxx.png')}} .....>



import sqlite3 

connection = sqlite3.connect("face.db",check_same_thread=False)

cursor = connection.cursor()


cursor.execute(



"""    CREATE TABLE FACE(

face BLOB

)                         """



)
connection.commit()
cursor.close()
connection.close()
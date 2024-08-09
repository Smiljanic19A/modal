from sshtunnel import SSHTunnelForwarder
import mysql.connector

# SSH tunnel settings
ssh_host = '142.132.147.131'
ssh_port = 22  
ssh_username = 'mysqltunnel'  
ssh_password = 'uH3iN0cP3hZ5kH3o' 

# MySQL settings
mysql_host = '127.0.0.1'  # The MySQL server's IP address (localhost because we're using an SSH tunnel)
mysql_port = 3306  # The MySQL server's port (default is 3306)
mysql_username = 'root'  # MySQL username
mysql_password = 'Gp570402BG'  # MySQL password
mysql_db = 'qira'  # MySQL database name

# Create the SSH tunnel
with SSHTunnelForwarder(
    (ssh_host, ssh_port),
    ssh_username=ssh_username,
    ssh_password=ssh_password,
    remote_bind_address=(mysql_host, mysql_port)
) as tunnel:

    # Connect to MySQL
    conn = mysql.connector.connect(
        host=mysql_host,
        port=tunnel.local_bind_port,
        user=mysql_username,
        password=mysql_password,
        database=mysql_db
    )

    # Do some MySQL operations here
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES;")
    for x in cursor:
        print(x)

    # Close MySQL connection
    cursor.close()
    conn.close()

openssl req -x509 -nodes -days 3650 -newkey rsa:2048 -keyout nginx-server.key -out nginx-server.crt -config openssl.conf

# Needs to be done on one of the plasma servers

import pyuda
pyuda.Client.server = "data.mastu.ukaea.uk"
pyuda.Client.port = 56565
client = pyuda.Client()
ip = client.get("ip", "12345")
print(ip.data)
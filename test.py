import urllib
import urllib.request
# If you are using Python 3+, import urllib instead of urllib2

import json 


data =  {

        "Inputs": {

                "input1":
                {
                    "ColumnNames": ["Col1", "Col2"],
                    "Values": [ [ "value", "value" ], [ "value", "value" ], ]
                },        },
            "GlobalParameters": {
}
    }

body = str.encode(json.dumps(data))

url = 'https://ussouthcentral.services.azureml.net/workspaces/0c025ac58964406d8f41281fc76f62f6/services/eaa7db383038462dbe1c03efd94433ae/execute?api-version=2.0&details=true'
api_key = 'PQXe70W/u2kCqguhih1yE2g7JDLhgfUnshr7nOdMM52s/MsEqNTKm7MpDABHqQc0DEmNewk3Z7dwiVfiE3c7Uw=='
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers) 

try:
    response = urllib.request.urlopen(req)
    response = urllib.request.urlopen(req)
    result = response.read()
    print(result) 

except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    print(error.info())
    print(json.loads(error.read()))                 
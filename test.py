import urllib
import urllib.request
# If you are using Python 3+, import urllib instead of urllib2

import json 


data =  {

        "Inputs": {

                "input1":
                {
                    "ColumnNames": ["Col1", "Col2"],
                    "Values": [ [ "Message","Hey, see you tomorrow?" ], 
                               [ "Message","URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"], ]
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
    html_response = response.read()
    encoding = response.headers.get_content_charset('utf-8')
    decoded_html = html_response.decode(encoding)
    occurances = [i for i in range(len(decoded_html)) if decoded_html.startswith('0.', i)]
    i=0
    for occurance in occurances:
        i+=1 
        print(f"Probability {i}. message is spam: {decoded_html[occurance:occurance+15]}%")


except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    print(error.info())
    print(json.loads(error.read()))                 
    
    
    
    
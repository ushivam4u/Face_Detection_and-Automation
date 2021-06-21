from twilio.rest import Client

def whatsapp():
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN'] 
    client = Client(account_sid, auth_token) 
    ss=['whatsapp:+9191050xxxxx','whatsapp:+919760xxxxxx','whatsapp:+918961xxxxxx','whatsapp:+918276xxxxxx','whatsapp:+918373xxxxxx']
    for i in ss:
        message = client.messages.create( 
                              from_='whatsapp:+14155238886',  
                              body='A warm welcome from Tech-Trollers!! Have a good day',      
                              to= i 
                          ) 
        print(message.sid)

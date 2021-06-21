from twilio.rest import Client

def whatsapp():
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN'] 
    client = Client(account_sid, auth_token) 
    ss=['whatsapp:+919105088479','whatsapp:+919760105131','whatsapp:+918961881014','whatsapp:+918276972706','whatsapp:+918373969216']
    for i in ss:
        message = client.messages.create( 
                              from_='whatsapp:+14155238886',  
                              body='A warm welcome from Tech-Trollers!! Have a good day',      
                              to= i 
                          ) 
        print(message.sid)

import smtplib
import os
from smtplib import SMTPException


def send_mail():
    password= os.environ['mail_pass']
    # set up the SMTP server
    s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    #s.starttls()
    s.login('ushivam4u@gmail.com' , password)
    try:
        sender = 'ushivam4u@gmail.com'
        receivers = ["Shubham.upload@gmail.com","rahulkeshari600@gmail.com","jainshubhangini@gmail.com","Khannahimangini@gmail.com"]
        message = """From: Python Greetings Program 
To: Owner
Subject: Greetings!!
Warm Welcome from Tech-Trollers!!
"""
        
        for i  in receivers:
            s.sendmail(sender, i , message)         
            print("Successfully sent email to -->"+ i)
    except Exception as E:
        print("Error: unable to send email\n")

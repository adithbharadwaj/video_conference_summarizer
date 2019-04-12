import smtplib
import datetime
import pandas as pd
import re
from email.mime.text import MIMEText

def get_email_list(path):
        dataframe = pd.read_excel(path)

        match_list = []

        for item in dataframe:
            match_list=dataframe["email addresses"].tolist()

        print(match_list)

        if(len(match_list)>0):
                return match_list

#module to send mail to receipient list
def send_mail(recipients, content, From = "ByteMe.RedHat@gmail.com"):

    mail = smtplib.SMTP('smtp.gmail.com',587)
    mail.ehlo()
    mail.starttls()
    myMailId = From
    myPassword = "3stickersneeded@"
    mail.login(myMailId, myPassword)

    msg = MIMEText(content)
    msg['Subject'] = "Video Conference Summary"
    msg['From'] = From
    msg['To'] = ", ".join(recipients)

    mail.sendmail(myMailId,recipients,msg.as_string())
    mail.close()

def send():
    path = 'demo.xlsx'
    recipients = get_email_list(path)
    current = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    f = open('summary.txt', 'r')
    summary = f.read()
    content = '\n' + current + '\n' + summary
    print(content)
    send_mail(recipients, content)
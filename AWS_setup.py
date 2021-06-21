import os
import subprocess as sp
import time

def aws_setup():
    print(" Launching EC2 instance on your AWS ") 
    os.system("aws ec2 run-instances --image-id ami-026669ec456129a70 --tag-specifications ResourceType=instance,Tags=[{Key=Name,Value=Tech-Trollers-Instance}] --instance-type t2.micro --security-group-ids sg-0075f1a55914516cc --key-name AWSkey")
    print(" Instance Launched")
    print("You can also see from the WEB UI that the Amazon Linux has been launched")
    print("Launching and attaching EBS volume of 5GB to the instance")
    os.system("aws ec2 create-volume --availability-zone ap-south-1a --volume-type gp2 --size 5 --tag-specifications ResourceType=volume,Tags=[{Key=Name,Value=Tech-Trollers-EBS}] ")
    print(" EBS Volume Launched")
    time.sleep(20)
    volid=sp.getoutput('aws ec2 describe-volumes  --filters Name=size,Values=5 --query "Volumes[*].VolumeId" --output=text')
    instid=sp.getoutput('aws ec2 describe-instances --filters Name=instance-state-name,Values=running --query "Reservations[*].Instances[*].InstanceId" --output=text')
    print(" Attaching volume to the instance")
    os.system("aws ec2 attach-volume --volume-id {} --instance-id {} --device /dev/sdf".format(volid,instid))
    print("EBS volume has been created and attached. You can see it from the web UI")

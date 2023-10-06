import sys
import paramiko
import time
import os

from googleapiclient import discovery
from google.oauth2.service_account import Credentials


def authenticate_vm(path):
    credentials = Credentials.from_service_account_file(path)
    return discovery.build('compute', 'v1', credentials=credentials)

def _start_ssh_session(compute, creds, username, passphrase):
    response = compute.instances().get(project="gpu-insatnce", zone='us-central1-a', instance='demos-tests').execute()
    external_ip = response['networkInterfaces'][0]['accessConfigs'][0]['natIP']

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(
        external_ip,
        username=username,
        key_filename=creds,
        passphrase=passphrase,
    )

    # Execute the command on the instance
    stdin, stdout, stderr = ssh.exec_command('cd actions-runner; nohup ./run.sh')

    # Read the output of the command
    output = stdout.read().decode()

    # Close the SSH connection
    ssh.close()

    return output

def start_runner(creds, ssh_creds, ssh_user, key_passphrase, id="gpu-insatnce", zone='us-central1-a',
                 instance='demos-tests'):

    compute = authenticate_vm(creds)
    request = compute.instances().start(project=id, zone=zone, instance=instance)
    request.execute()
    time.sleep(60)

    _start_ssh_session(compute, ssh_creds, ssh_user, key_passphrase)


def stop_runner(creds):
    compute = authenticate_vm(creds)
    request = compute.instances().start(project="gpu-insatnce", zone='us-central1-a', instance='demos-tests')
    request.execute()


if __name__ == "__main__":
    ssh_user, key_passphrase, stop_vm = sys.argv[1], sys.argv[2], sys.argv[3]
    gcp_credentials = 'gcp_auth.json'
    ssh_credentials = '~/.ssh/id_rsa'

    if stop_vm == "true":
        # Stop the instance
        stop_runner(gcp_credentials)
    else:
        # Start the instance
        ssh_key_path = os.path.expanduser()
        start_runner(gcp_credentials, ssh_credentials, ssh_user, key_passphrase)

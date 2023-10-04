import sys
import json
import paramiko
import io

from google.auth import compute_engine
from googleapiclient import discovery
from google.oauth2.service_account import Credentials
from google.auth import impersonated_credentials

def authenticate_vm(path):
    credentials = Credentials.from_service_account_file(path)
    return discovery.build('compute', 'v1', credentials=credentials), credentials
def start_runner(creds, key, id = "gpu-insatnce", zone='us-central1-a', instance='demos-tests'):
    compute, credentials = authenticate_vm(creds)
    compute.instances().start(project=id, zone=zone, instance=instance).execute()
    # request = compute.instances().get(project=id, zone=zone, instance=instance)
    # response = request.execute()
    #
    # # Extract the external IP address of the instance
    # external_ip = response['networkInterfaces'][0]['accessConfigs'][0]['natIP']


    # Get the SSH username (assuming it's stored in the credentials)
    ssh_username = credentials.service_account_email
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=f'{instance}.{zone}.compute.internal',
        username=ssh_username,  # Typically 'your-username' or 'gce-username'
        pkey = paramiko.RSAKey(file_obj=io.StringIO(key)),
    )


    # Execute the command on the instance
    stdin, stdout, stderr = ssh.exec_command('cd actions-runner; nohup ./run.sh')

    # Read the output of the command
    output = stdout.read().decode()

    # Close the SSH connection
    ssh.close()

    return output

if __name__ == "__main__":
    key = sys.argv[1]
    # Start the instance
    start_runner('gcp_auth.json', key)




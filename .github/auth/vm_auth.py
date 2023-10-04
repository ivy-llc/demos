import sys
import json
import paramiko
import time
import io

from google.auth import compute_engine
from googleapiclient import discovery
from google.oauth2.service_account import Credentials
from google.auth import impersonated_credentials

def authenticate_vm(path):
    credentials = Credentials.from_service_account_file(path)
    return discovery.build('compute', 'v1', credentials=credentials)
def start_runner(creds, key, ssh_username, id = "gpu-insatnce", zone='us-central1-a', instance='demos-tests'):
    compute = authenticate_vm(creds)
    request = compute.instances().start(project=id, zone=zone, instance=instance)
    request.execute()
    time.sleep(60)

    response = compute.instances().get(project=id, zone=zone, instance=instance).execute()
    external_ip = response['networkInterfaces'][0]['accessConfigs'][0]['natIP']
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        external_ip,
        username=ssh_username,  # Typically 'your-username' or 'gce-username'
        pkey=paramiko.RSAKey(file_obj=io.StringIO(key)),
    )


    # Execute the command on the instance
    stdin, stdout, stderr = ssh.exec_command('cd actions-runner; nohup ./run.sh')

    # Read the output of the command
    output = stdout.read().decode()

    # Close the SSH connection
    ssh.close()

    return output

if __name__ == "__main__":
    key, username = sys.argv[1], sys.argv[2]
    # Start the instance
    start_runner('gcp_auth.json', str(key), username)




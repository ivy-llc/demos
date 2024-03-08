# Contributor Demos - README

Welcome to the Contributor Demos section of our `unifyai/demos` repository! This folder contains the exciting ML demos that our community created using `ivy`. Below, you will find detailed instructions on how to getting involved in these projects, ranging from plant disease recognition to sarcasm detection, and everything in between.

## Important Notes

- **Single PR Acceptance:** For each issue, only one PR will be merged. We encourage collaboration and communication within the community to combine efforts and submit the best possible solution.
- **Rising Contributors:** We're happy to welcome the contributors of the merged PR as Rising contributors [Rising Contributors](https://unify.ai/docs/ivy/overview/contributing/volunteer_program.html#rising-contributor) in our community. This status comes with recognition within our community and opportunities for further involvement in project development.

## Getting Started

### Step 1: Fill Out the Contributor Form

Before diving into the code, we ask all interested contributors to fill out a short form. This form captures essential information such as your name, email, whether you plan to contribute individually or as part of a group, and the project are you willing to work on. Please complete the form at this link: [Contributor Form](https://forms.gle/kTuRAmLh2SzKmm146). Informations of all the projects are provided in the [Github Projects](https://github.com/orgs/unifyai/projects/18). Each project comes with a unique set of challenges and learning opportunities.

### Step 2: Setting up Ivy

Once you've filled out the form, you're ready to start contributing to the project of your choice. To get started, follow these steps:
1. Fork the [unifyai/demos](https://github.com/unifyai/demos) repository to your GitHub account.
2. Clone the forked repository to your local machine using the following command:
   ```bash
   git clone <your-forked-repo-url>
   ```
3. Navigate to the `demos` directory:
   ```bash
    cd demos
    ```
4. Create a new branch for your work:
    ```bash
    git checkout -b <branch-name>
    ```
5. Create a virtual environment and install ivy(Alternatively, you can set up ivy in your local machine using the [official documentation](https://unify.ai/docs/ivy/overview/contributing/setting_up.html)):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install ivy
    ```
6. Generating API key:
    - Navigate to [UnifyAI Console](https://console.unify.ai)
    - Sign up/Log in to your account and create a new key for `Ivy Transpiler`
7. Create a new directory `.ivy` in the root of the project and create a file `key.pem` inside the `.ivy` directory.
8. Paste the API key in the `key.pem` file and save it.
If you have any issues with the setup, please feel free to ask for help on our [discord server](https://discord.gg/sg2QRAvY).

### Step 3: Solving an Issue

Start working on the issue you've chosen. Create a jupyter notebook under the directory of the project you're working on. You can also have the trained model in the same directory.

### Step 4: Submitting Your Pull Request (PR)

After developing your solution, it's time to submit a Pull Request (PR) to the `unifyai/demos` repository. Make sure to link the specific issue you're addressing by including `closes <#issue no>.` in your PR description. This notation will help us track which PRs aim to solve which issues.

#### PR Description Guidelines

To streamline the review process and facilitate badge awarding, we require a uniform format for PR descriptions. Please include the GitHub IDs of all team members involved in the project at the end of your PR description, following this template:

```
# Team Members
- GitHubID1
- GitHubID2
- GitHubID3
- GitHubID4 (if applicable)
```

## Contribution Guidelines

- Ensure your code is well-documented and follows best practices for readability and maintainability.
- Be creative! We value innovation and unique approaches to solving problems.
- Stay engaged. Participate in discussions on your PR for feedback, suggestions, and improvements.

## Questions or Concerns?

If you have any questions or need clarification on the process, please feel free to ask us on our [discord server](https://discord.gg/sg2QRAvY). We're here to support you throughout your contribution journey.

## Let's Build Together!

Your contributions play a vital role in the success of this initiative. By participating, you're not only enhancing your skills but also contributing to a project that could have a real-world impact. We're excited to see your innovative solutions and to welcome you as a valuable member of our growing community.

## Video

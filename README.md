# Medical Claim Validation using LangGraph and OpenAI API.

### Goals for this project

Need to build a FastAPI based backend API service that can

-   Take in multiple PDF files (bill, discharge_summary, id_card, pharmacy_bill, claim_form, other)
-   (Extract) Extract text using LLM
-   (Classify) Process and classify them using LLM agents
-   (Validate) Perform validation checks \- missing docs/cross validation \- (name, date, amount mismatches)
-   Produce a final decision: approved / rejected / manual_review
-   Return full consolidated JSON

### Advanced solution: ([discussed here](#why-i-didnt-use-agents))

Use multi-agent workflow to analyze and format extracted text from each class of PDF files, by separate agents.

-   BillAgent
-   DischargeAgent
-   IDAgent
-   PharmacyAgent (Optional)

### Setup and Run

-   Make sure Python3.12 or higher is installed.
-   Install the dependencies:

    ```bash
    uv sync
    ```

-   Set the environment variables for OpenAI API key (`.env`):

    ```bash
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

-   Run the FastAPI server:

    ```bash
    uvicorn main:app --reload
    ```

-   Access the API documentation at `http://localhost:8000/docs`

---

### What each of the documents will contain (assumed) ?

-   id_card:
    -   member_id
    -   policy_number
    -   coverage_details
-   claim_form:
    -   claim_number
    -   diagnostis_codes
    -   procedure_codes
    -   service_dates
    -   total_claimed_amount
-   discharge_summary:
    -   diagnosis
    -   procedures_performed
    -   admission_date
    -   discharge_date
    -   doctor_name
-   bill:
    -   amount
    -   date
    -   line_items
    -   diagnosis_codes
    -   provider
-   pharmacy_bill:
    -   amount
    -   date
    -   medicines
    -   patient_name

**Mandatory documents:**

1. **claim_form** \- Without this, there's no formal claim request (what are you approving?)
2. **At least one bill** (bill OR pharmacy_bill) \- Need proof of expense to reimburse
3. **id_card** \- Proves person is covered under policy

### Thought process

-   Build a FastAPI backend
-   `POST /process-claim` takes in multiple PDF files
-   PDF files converted to images (PDF to images node)
    -   Each PDF file can have multiple pages \- use the first few pages for classifying the type of pdf
    -   store the file details in a dataclass (typed class for easy access)
-   Upload PDF files to llm \- base64 encoded and extract the details structured and classified file type. (extract data node)
-   Use the structured data extracted in previous node to validate \- upload the extracted data to LLM and validate via prompting (validate node)
-   Finally based on the last step’s output decide the final state of the claim (decide node)
-   Return the response received from the graph invocation as a dict.

### Why I didn’t use agents?

I think agents are useful when we have to dynamically decide which tools to use based on parsed data of a file.  
Mostly when the steps are not clear, and must be decided based on the contents of the file.  
Since, in this case steps are clear we can proceed with simple LangGraph nodes.

### Why I didn’t extract text from PDF?

Text extract from PDF and storing them as embeddings is only useful when the contents of PDF are mostly plain-text and do not need to obey any formatting.  
Here, in case of medical bills often output is in tabular format, so I think using LLMs or any good OCR (even offline) should be used to extract text.  
The final decision can be made ofcourse after multiple test runs.  
**What I did?** I have extracted pages from PDF and uploaded them as images for getting all the details extracted using OCR tool (openai vision API).  
I have provided expected format using “few shot” prompting style to get the output in desired JSON format.

### Testing

Testing this source code wasn’t possible, since I do not have the relevant files. I am open to test this code out and possibly fix any bugs.

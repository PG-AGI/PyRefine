"""
Test file with long strings to demonstrate string_fixer functionality.
"""

# Long normal string
error_message = "This is a very long error message that exceeds the maximum " \
                "line length and should be split into multiple lines " \
                "automatically"

# Long f-string
user_id = 12345
username = "john_doe"
ip_address = "192.168.1.100"
login_message = f"User {username} with ID {user_id} successfully logged in " \
                f"from IP address {ip_address} at this timestamp"

# Long raw string (Windows path)
config_path = r"C:\Users\Administrator\Documents\MyProjects\PythonApplication" \
              r"\Configuration\Settings\Production\database_config.ini"

# Long SQL query
query = "SELECT users.id, users.name, users.email, orders.total FROM users " \
        "INNER JOIN orders ON users.id = orders.user_id WHERE orders.status = " \
        "'completed'"

# Long API URL
api_endpoint = "https://api.example.com/v2/users/profile/settings/notification" \
               "s/preferences/email?include_metadata=true&detailed=true&format" \
               "=json"


# Function with long docstring
def process_user_data(user_id, username, email):
    """This function processes user data by validating the input parameters,
    checking database constraints, transforming data formats, and persisting
    to the database."""
    pass


# Multiple long strings in one block
log_info = "Processing batch job with ID 98765 - this operation will take " \
           "several minutes to complete and may consume significant resources"
log_warning = "Database connection pool is running low on available " \
              "connections - consider increasing the pool size in " \
              "configuration"
log_error = "Failed to connect to external API service at " \
            "https://external-service.example.com/api/v1/endpoint after 3 " \
            "retry attempts"


# Complex f-string with expressions
def calculate_stats(data):
    result = f"Analysis complete: Mean={sum(data)/len(data)}, " \
             f"Min={min(data)}, Max={max(data)}, Total items " \
             f"processed={len(data)} from dataset"
    return result


# Nested string with escape sequences
json_template = '{"user_id": "12345", "username": "john", "email": ' \
                '"john@example.com", "preferences": {"notifications": true, ' \
                '"theme": "dark"}}'

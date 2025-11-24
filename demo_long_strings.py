"""
Demo file showing various string types that need fixing.

This file intentionally has long strings to demonstrate string_fixer functionality.
"""

# Long normal string
error_message = "This is a very long error message that exceeds the maximum line length and should be split into multiple lines automatically"

# Long f-string
user_id = 12345
username = "john_doe"
ip_address = "192.168.1.100"
login_message = f"User {username} with ID {user_id} successfully logged in from IP address {ip_address} at this timestamp"

# Long raw string (Windows path)
config_path = r"C:\Users\Administrator\Documents\MyProjects\PythonApplication\Configuration\Settings\Production\database_config.ini"

# Long SQL query
query = "SELECT users.id, users.name, users.email, orders.total FROM users INNER JOIN orders ON users.id = orders.user_id WHERE orders.status = 'completed'"

# Long API URL
api_endpoint = "https://api.example.com/v2/users/profile/settings/notifications/preferences/email?include_metadata=true&detailed=true&format=json"


# Function with long docstring
def process_user_data(user_id, username, email):
    """This function processes user data by validating the input parameters, checking database constraints, transforming data formats, and persisting to the database."""
    pass


# Multiple long strings in one block
log_info = "Processing batch job with ID 98765 - this operation will take several minutes to complete and may consume significant resources"
log_warning = "Database connection pool is running low on available connections - consider increasing the pool size in configuration"
log_error = "Failed to connect to external API service at https://external-service.example.com/api/v1/endpoint after 3 retry attempts"


# Complex f-string with expressions
def calculate_stats(data):
    result = f"Analysis complete: Mean={sum(data)/len(data)}, Min={min(data)}, Max={max(data)}, Total items processed={len(data)} from dataset"
    return result


# Nested string with escape sequences
json_template = '{"user_id": "12345", "username": "john", "email": "john@example.com", "preferences": {"notifications": true, "theme": "dark"}}'


# =============================================================================
# EXAMPLES WHERE STRING FIXER WILL NOT WORK
# =============================================================================

# 1. COMMENTS - String fixer only handles strings, not comments
# This is a very long comment that exceeds the maximum line length and will not be split by the string fixer because it's not a string literal

# 2. IMPORT STATEMENTS - Long imports won't be split
# from some_very_long_module_name_that_exceeds_the_maximum_line_length_and_should_not_be_split import some_function, another_function, yet_another_function

# 3. FUNCTION DEFINITIONS - Long function signatures won't be split
# def some_function_with_many_parameters(parameter_one, parameter_two, parameter_three, parameter_four, parameter_five, parameter_six):
#     pass

# 4. CLASS DEFINITIONS - Long class inheritance won't be split
# class SomeClassWithVeryLongInheritance(BaseClassOne, BaseClassTwo, BaseClassThree, BaseClassFour, BaseClassFive):
#     pass

# 5. VARIABLE ASSIGNMENTS - Long variable names won't be split
some_very_long_variable_name_that_exceeds_the_maximum_line_length = (
    "short " "value"
)

# 6. ALREADY PROPERLY SPLIT MULTI-LINE STRINGS - Won't be touched
already_split = (
    "This string is already properly split using parentheses "
    "and won't be modified by the string fixer"
)

# 7. SHORT STRINGS - Strings within line limit won't be changed
short = "This is short enough"

# 8. URLs THAT SHOULDN'T BE BROKEN - May break semantic meaning
# (String fixer might split this, but it could break the URL functionality)
important_url = (
    "https://api.example.com/very/long/path/that/should/not/be/bro"
    "ken/at/bad/places/because/it/would/break/the/api/call"
)

# 9. CODE LOGIC - Non-string code won't be touched
# if some_condition and another_condition and yet_another_condition and one_more_condition:
#     do_something()

# 10. DICTIONARY/SET LITERALS - Won't be split
my_dict = {
    "key1": "value1",
    "key2": "value2",
    "key3": "value3",
    "key4": "valu" "e4",
    "key5": "value5",
}

# 11. LIST LITERALS - Won't be split
my_list = [
    "item1",
    "item2",
    "item3",
    "item4",
    "item5",
    "item6",
    "item7",
    "ite" "m8",
]

# 12. COMPLEX EXPRESSIONS - Won't be split
# result = calculate_something() + process_data() + validate_input() + save_to_database()

# 13. ASSERT STATEMENTS - Won't be split
# assert condition_one and condition_two and condition_three and condition_four, "All conditions must be true"

# 14. WITH STATEMENTS - Won't be split
# with open_file() as f1, open_another_file() as f2, open_third_file() as f3:
#     process_files(f1, f2, f3)

# 15. EXCEPTION HANDLING - Won't be split
# try:
#     risky_operation()
# except (ExceptionType1, ExceptionType2, ExceptionType3, ExceptionType4) as e:
#     handle_error(e)

# 16. DECORATORS - Won't be split
# @some_decorator_with_many_parameters(param1="value1", param2="value2", param3="value3")
# def decorated_function():
#     pass

# 17. LAMBDA EXPRESSIONS - Won't be split
# lambda_function = lambda x, y, z, a, b, c: x + y + z + a + b + c

# 18. TERNARY OPERATORS - Won't be split
# result = "success" if condition_one and condition_two and condition_three else "failure"

# 19. STRING CONCATENATION - Already split manually
manual_concat = (
    "This is "
    + "manually "
    + "concatenated "
    + "and "
    + "won't "
    + "be "
    + "touched"
)

# 20. F-STRINGS WITH COMPLEX NESTING - May not handle perfectly
# complex_fstring = f"Result: {get_data()['users'][0]['profile']['settings']['preferences']['theme']}"

# 21. MULTI-LINE STRINGS WITH TRIPLE QUOTES - Already properly formatted
multi_line_doc = """
This is a multi-line string that is already properly formatted
and won't be touched by the string fixer.
"""

# 22. REGULAR EXPRESSIONS - May not split optimally
# regex_pattern = r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"

# 23. SQL QUERIES ALREADY SPLIT - Won't be touched
existing_sql = (
    "SELECT users.id, users.name FROM users " "WHERE users.active = 1"
)

# 24. CONFIG STRINGS - May break configuration
# config_line = "database_url=mysql://user:password@host:port/database?charset=utf8&autocommit=true"

# 25. BINARY DATA STRINGS - May not be appropriate to split
# binary_data = "xff\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07\\x08\\x09\\x0a\\x0b\\x0c\\x0d\\x0e\\x0f" * 10

# Search system -> perception: targets not found yet in the running command.
# Published when the set changes, latched (transient local, depth 1).
# Empty when no command is searching: perception uses its own vocabulary.
std_msgs/Header header        # stamp: when the set changed
int32 command_id              # increases with every command (resume keeps it)
SearchTarget[] targets

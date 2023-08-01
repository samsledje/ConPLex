import functools

import flask


def user_required(view):
    @functools.wraps(view)
    def decorated_view(*args, **kwargs):
        if flask.g.user is None:
            return flask.abort(401)

        return view(*args, **kwargs)

    return decorated_view

# coding=utf-8

from contextlib import contextmanager

from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


Base = declarative_base()
Meta = MetaData()


def build_table(table, renew=False, eng=None, verbose=False):
    if eng is None:
        # eng = EnginePointer.get()
        raise NotImplementedError

    table_info = f'{table.__table_args__["schema"]}.{table.__tablename__}'

    if table.__table__.exists(eng) and not renew:
        if verbose:
            print(f'{table_info} exits.')
    elif table.__table__.exists(eng) and renew:
        if verbose:
            print(f'{table_info} exits.')
        table.__table__.drop(eng)

        if verbose:
            print(f'{table_info} dropped.')
        table.__table__.create(eng)

        if verbose:
            print(f'{table_info} created.')
    else:
        table.__table__.create(eng)
        if verbose:
            print(f'{table_info} created.')


@contextmanager
def session_scope(eng=None):
    if eng is None:
        raise NotImplementedError
        # eng = EnginePointer.get()
    session = Session(bind=eng)
    try:
        yield session
        session.commit()
    except Exception as _:
        session.rollback()
        raise
    finally:
        session.close()


def truncate_table(table, eng=None):
    if eng is None:
        # session = sessionmaker(EnginePointer.get())()
        raise NotImplementedError
    else:
        session = sessionmaker(eng)()
    session.query(table).delete(synchronize_session='fetch')
    session.commit()
    session.close()

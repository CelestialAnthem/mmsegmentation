import oss2
import os
import tos
import datetime
import hashlib
import hmac

from oss2.exceptions import RequestError

# oss配置 和 tos配置
OSS_CONFIG_DICT = {
    "OSS_ENV": "TOS",
    "OSS": {
        "access_key_id": "",  #补齐ak
        "access_key_secret": "", #补齐sk
        "internal_end_point": "oss-cn-beijing-internal.aliyuncs.com",
        "external_end_point": "oss-cn-beijing.aliyuncs.com",
        "role_arn": "acs:ram::1885092470293468:role/lucas-data"
    },
    "TOS": {
        "access_key_id": os.getenv("HAOMO_AK"),
        "access_key_secret": os.getenv("HAOMO_SK"),
        "internal_end_point": "tos-cn-beijing.ivolces.com",
        "external_end_point": "tos-cn-beijing.volces.com",
        "region": "cn-beijing",
        "role_arn": "trn:iam::2100219150:role/STS-lucss-data-send"
    }
}


class BaseOsService(object):
    """"
        os存储操作基类
        1、所继承的类，类中的方法需要保持一致性
        2、relative_path方法为了适配之前的服务调用方式，返回的操作的oss对象类
    """
    _OSS_CONFIG_DICT = None

    # 指定bocket从阿里云拉取数据列表
    ALI_BUCKET_NAME_LIST = []

    # 自适应列表
    AUTO_BUCKET_NAME_LIST = []

    TOS_PREFIX = "tos"
    OSS_PREFIX = "oss"

    TOS_ENV = "TOS"
    OSS_ENV = "OSS"
    AUTO_ENV = "AUTO"

    POLICY_TEST = '{"Version": "1", "Statement": [{"Action": ["oss:PutObject", "oss:GetObject"], ' \
                  '"Effect": "Allow", "Resource": ["acs:oss:*:*:%s/*"]}]}'

    @classmethod
    def new(cls, config_dict):
        if config_dict is not None:
            cls._OSS_CONFIG_DICT = config_dict
        return cls._OSS_CONFIG_DICT

    def __init__(self, path, client, oss_prefix):
        """
        param path: string, input oss path, example: /test/test.json
        """
        self.client = client
        self.path = path
        self.oss_prefix = oss_prefix
        self._, self.bucket_name, self.relative_path = self.parse_parse(path)

    def __exit__(self):
        self.close()

    def sign(self, key, msg):
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

    def getSignatureKey(self, key, dateStamp, regionName, serviceName):
        kDate = self.sign(key.encode('utf-8'), dateStamp)
        kRegion = self.sign(kDate, regionName)
        kService = self.sign(kRegion, serviceName)
        kSigning = self.sign(kService, 'request')
        return kSigning

    def getSignHeaders(self, method, service, host, region, request_parameters, access_key, secret_key):
        contenttype = 'application/x-www-form-urlencoded'
        accept = 'application/json'
        t = datetime.datetime.utcnow()
        xdate = t.strftime('%Y%m%dT%H%M%SZ')
        datestamp = t.strftime('%Y%m%d')
        # *************  1: 拼接规范请求串*************
        canonical_uri = '/'
        canonical_querystring = request_parameters
        canonical_headers = 'content-type:' + contenttype + '\n' + 'host:' + host + '\n' + 'x-date:' + xdate + '\n'
        signed_headers = 'content-type;host;x-date'
        payload_hash = hashlib.sha256(('').encode('utf-8')).hexdigest()
        canonical_request = method + '\n' + canonical_uri + '\n' + canonical_querystring + '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash
        # *************  2：拼接待签名字符串*************
        algorithm = 'HMAC-SHA256'
        credential_scope = datestamp + '/' + region + '/' + service + '/' + 'request'
        string_to_sign = algorithm + '\n' + xdate + '\n' + credential_scope + '\n' + hashlib.sha256(
            canonical_request.encode('utf-8')).hexdigest()
        # *************  3：计算签名 *************
        signing_key = self.getSignatureKey(secret_key, datestamp, region, service)
        signature = hmac.new(signing_key, (string_to_sign).encode('utf-8'), hashlib.sha256).hexdigest()
        # *************  4：添加签名到请求header中 *************
        authorization_header = algorithm + ' ' + 'Credential=' + access_key + '/' + credential_scope + ', ' + 'SignedHeaders=' + signed_headers + ', ' + 'Signature=' + signature
        headers = {'Accpet': accept, 'Content-Type': contenttype, 'X-Date': xdate,
                   'Authorization': authorization_header}
        return headers

    def close(self):
        pass

    def sign_url(self, *args, **kwargs):
        raise NotImplementedError

    def sign_put_url(self, *args, **kwargs):
        raise NotImplementedError

    def out_url(self, url):
        raise NotImplementedError

    def download_file(self):
        raise NotImplementedError

    def put_object(self, *args, **kwargs):
        raise NotImplementedError

    def gen_upload_token(self, userid, expires=1000):
        raise NotImplementedError

    @classmethod
    def parse_parse(cls, path):
        """解析完整oss/tos地址:
        path_example:
            oss://haomo-airflow/test/test.json
        return:
            oss_prefix: oss/tos环境 example: oss or tos
            bucket_name: 云存储桶名称 example: haomo-airflow
            relative_path: 相对路径 example: test/test.json
        """
        seq = "://"
        oss_prefix, _path = path.split(seq)
        bucket_name = _path.split("/")[0]
        relative_path = path[len(bucket_name) + len(oss_prefix) + len(seq) + 1:]
        return oss_prefix, bucket_name, relative_path

    @classmethod
    def init_oss(cls, path, oss_env=None):
        oss_env = oss_env or cls._OSS_CONFIG_DICT["OSS_ENV"]

        # 若是oss采集桶，则强制使用oss客户端
        oss_prefix, bucket_name, relative_path = cls.parse_parse(path)

        if oss_prefix == cls.OSS_PREFIX and bucket_name in cls.ALI_BUCKET_NAME_LIST:
            oss_env = cls.OSS_ENV
        elif bucket_name in cls.AUTO_BUCKET_NAME_LIST:
            oss_env = cls.AUTO_ENV

        if oss_env == cls.TOS_ENV:
            return TosService.init(path)
        elif oss_env == cls.OSS_ENV:
            return OssService.init(path)
        else:
            # 自动使用oss 环境
            if path.startswith(cls.TOS_PREFIX):
                return TosService.init(path)
            return OssService.init(path)


class TosService(BaseOsService):

    def close(self):
        """关闭客户端长连接"""
        self.client.close()

    @classmethod
    def init(cls, path, is_internal=True):
        """处理相对路径"""
        # 判断是否以/开头
        # 实例化客户端
        endpoint = cls._OSS_CONFIG_DICT[cls.TOS_ENV]["internal_end_point"] if \
            is_internal else cls._OSS_CONFIG_DICT[cls.TOS_ENV]["external_end_point"]

        client = tos.TosClientV2(
            ak=cls._OSS_CONFIG_DICT[cls.TOS_ENV]["access_key_id"],
            sk=cls._OSS_CONFIG_DICT[cls.TOS_ENV]["access_key_secret"],
            endpoint=endpoint,
            region=cls._OSS_CONFIG_DICT[cls.TOS_ENV]["region"],
        )
        self = cls(path=path, client=client, oss_prefix=cls.TOS_PREFIX)
        return self

    def sign_url(self, expires=3600, params=None):
        """计算下载http"""
        out = self.client.pre_signed_url(
            tos.HttpMethodType.Http_Method_Get, bucket=self.bucket_name, key=self.relative_path, expires=expires)
        return out.signed_url

    def sign_put_url(self, expires=3600):
        """计算上传http"""
        out = self.client.pre_signed_url(
            tos.HttpMethodType.Http_Method_Put, bucket=self.bucket_name, key=self.relative_path, expires=expires)
        return out.signed_url

    def out_url(self, url, expires=3600):
        outer_self = TosService.init(path=self.path, is_internal=False)
        return outer_self.sign_url(expires=expires)

    def download_file(self):
        object_stream = self.client.get_object(self.bucket_name, self.relative_path)
        return object_stream.read()

    def get_object_to_file(self, local_path):
        return self.client.get_object_to_file(self.bucket_name, self.relative_path, local_path)

    def put_object(self, data, error_num=0):
        # obj = self.client.put_object(self.bucket_name, self.path, content=data)
        # return obj
        try:
            _ = self.client.put_object(self.bucket_name, self.relative_path, content=data)
            oss_path = os.path.join(self.oss_prefix + "://", self.bucket_name, self.relative_path)
            return oss_path
        except:
            """error_num 是上传oss 超时错误，默认允许重试3次"""
            if error_num >= 2:
                raise "error put object"
            self.client.put_object(self.relative_path, data, error_num=error_num + 1)

    def put_object_from_file(self, local_path):
        """上传本地文件"""
        return self.client.put_object_from_file(self.bucket_name, self.relative_path, local_path)

    def is_exists(self):
        try:
            self.client.head_object(self.bucket_name, self.relative_path)
            return True
        except tos.exceptions.TosServerError as e:
            if e.status_code == 404:
                return False
            raise e


class OssService(BaseOsService):

    @classmethod
    def init(cls, path):
        """处理相对路径"""
        oss_prefix, bucket_name, relative_path = cls.parse_parse(path)

        client = oss2.Bucket(
            auth=oss2.Auth(cls._OSS_CONFIG_DICT[cls.OSS_ENV]["access_key_id"],
                           cls._OSS_CONFIG_DICT[cls.OSS_ENV]["access_key_secret"]),
            endpoint=cls._OSS_CONFIG_DICT[cls.OSS_ENV]["internal_end_point"],
            bucket_name=bucket_name,
        )
        self = cls(path=path, client=client, oss_prefix=cls.OSS_PREFIX)
        return self

    def sign_url(self, expires=86400, params=None):
        """计算http签名地址"""
        if params:
            return self.client.sign_url("GET", self.relative_path, expires=expires, slash_safe=True, params=params)
        return self.client.sign_url("GET", self.relative_path, expires=expires, slash_safe=True)

    def sign_put_url(self, expires=3600):
        """计算http上传签名"""
        return self.client.sign_url('PUT', self.relative_path, expires, slash_safe=True)

    def out_url(self, url, expires=3600):
        url = url.replace(self._OSS_CONFIG_DICT[self.OSS_ENV]["internal_end_point"],
                          self._OSS_CONFIG_DICT[self.OSS_ENV]["external_end_point"])
        prefix = "http://"
        if url.startswith(prefix):
            url = "".join(["https://", url[len(prefix):]])
        return url

    def download_file(self):
        # 非相对路径需要处理成相对路径
        object_stream = self.client.get_object(self.relative_path)
        obj = object_stream.read()
        if object_stream.client_crc != object_stream.server_crc:
            print("The CRC checksum between client and server is inconsistent!")
            return None
        return obj

    def get_object_to_file(self, local_path):
        return self.client.get_object_to_file(self.relative_path, local_path)

    def put_object(self, data, error_num=0):
        """上传文件"""
        try:
            _ = self.client.put_object(self.relative_path, data)
            oss_path = os.path.join(self.oss_prefix + "://", self.bucket_name, self.relative_path)
            return oss_path
        except RequestError as error:
            """error_num 是上传oss 超时错误，默认允许重试3次"""
            if error_num >= 2:
                raise error
            self.client.put_object(self.relative_path, data, error_num=error_num + 1)

    def put_object_from_file(self, local_path):
        """上传本地文件"""
        return self.client.put_object_from_file(self.relative_path, local_path)

    def is_exists(self):
        return self.client.object_exists(self.relative_path)


class OssUpload(BaseOsService):

    @classmethod
    def relative_path(cls, path):
        # 实现单例
        BaseOsService.new(OSS_CONFIG_DICT)
        return cls.init_oss(path)


if __name__ == '__main__':

    oss_path = "oss://haomo-airflow/test/test.txt"
    tos_path = "tos://haomo-airflow/release/airflow/checker/2023-03-29/168007057205928400045769/card/1678193649085951.json"

    # 外层正常调用，不需要做任何操作
    ins = OssUpload.relative_path(tos_path)
    # 计算获取oss信息 http地址
    http_url = ins.sign_url()
    out_rul = ins.out_url(http_url)
    print(">>>>下载http地址", out_rul)
    # 计算上传http地址
    sign_put_url = ins.sign_put_url()
    print(">>>>上传http地址", sign_put_url)
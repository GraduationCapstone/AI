"""
테스트 스크립트 파일 저장 관리자

S3 또는 MinIO에 생성된 Playwright 테스트 파일을 저장하고
다운로드 URL을 반환합니다.
"""

import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class FileManager:
    """
    테스트 스크립트 파일 저장 관리자
    
    S3/MinIO에 Playwright 테스트 파일을 저장하고 관리합니다.
    
    Attributes:
        s3_client: boto3 S3 클라이언트
        bucket_name: S3 버킷 이름
        endpoint_url: S3 엔드포인트 (MinIO 사용 시)
    
    Example:
        >>> # S3 사용
        >>> manager = FileManager(
        ...     aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        ...     aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ...     bucket_name="probe-tests"
        ... )
        >>> 
        >>> # MinIO 사용
        >>> manager = FileManager(
        ...     endpoint_url="http://localhost:9000",
        ...     aws_access_key_id="minioadmin",
        ...     aws_secret_access_key="minioadmin",
        ...     bucket_name="probe-tests"
        ... )
        >>> 
        >>> # 파일 저장
        >>> result = manager.save_test_script(
        ...     test_code="import { test } from '@playwright/test'...",
        ...     repository_name="ecommerce-app",
        ...     test_name="login"
        ... )
        >>> print(result["download_url"])
    """
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
        bucket_name: str = "probe-test-scripts"
    ):
        """
        FileManager 초기화
        
        Args:
            endpoint_url: S3 엔드포인트 URL (MinIO: http://localhost:9000)
            aws_access_key_id: AWS Access Key ID
            aws_secret_access_key: AWS Secret Access Key
            region_name: AWS 리전 (S3만 해당)
            bucket_name: S3 버킷 이름
        
        Raises:
            ValueError: 필수 자격증명이 없는 경우
        """
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "AWS credentials are required. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )
        
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        
        # S3 클라이언트 생성
        client_config = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': region_name
        }
        
        # MinIO 사용 시 endpoint_url 추가
        if endpoint_url:
            client_config['endpoint_url'] = endpoint_url
        
        try:
            self.s3_client = boto3.client('s3', **client_config)
            logger.info(
                f"FileManager initialized: "
                f"endpoint={endpoint_url or 'AWS S3'}, "
                f"bucket={bucket_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
        
        # 버킷 존재 확인 및 생성
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """버킷이 없으면 생성"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # 버킷이 없으면 생성
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Bucket '{self.bucket_name}' created")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    raise
            else:
                logger.error(f"Error checking bucket: {e}")
                raise
    
    def save_test_script(
        self,
        test_code: str,
        repository_name: str,
        test_name: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        테스트 스크립트를 S3/MinIO에 저장
        
        Args:
            test_code: Playwright 테스트 코드
            repository_name: 저장소 이름
            test_name: 테스트 이름 (예: "login", "checkout")
            metadata: 추가 메타데이터 (선택)
        
        Returns:
            Dict: 저장 결과
                - filename: str (파일명)
                - storage_url: str (s3://bucket/key)
                - download_url: str (Pre-signed URL)
                - s3_key: str (S3 객체 키)
        
        Example:
            >>> result = manager.save_test_script(
            ...     test_code="import { test } ...",
            ...     repository_name="my-app",
            ...     test_name="login"
            ... )
            >>> print(result)
            {
                "filename": "login_20250109_123456.spec.js",
                "storage_url": "s3://probe-tests/my-app/login_20250109_123456.spec.js",
                "download_url": "https://...",
                "s3_key": "my-app/login_20250109_123456.spec.js"
            }
        """
        # 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.spec.js"
        
        # S3 키 생성 (저장소별 디렉토리 구조)
        s3_key = f"{repository_name}/{filename}"
        
        # 메타데이터 준비
        s3_metadata = metadata or {}
        s3_metadata.update({
            'repository': repository_name,
            'test_name': test_name,
            'created_at': timestamp
        })
        
        try:
            # S3에 업로드
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=test_code.encode('utf-8'),
                ContentType='application/javascript',
                Metadata={k: str(v) for k, v in s3_metadata.items()}
            )
            
            # Storage URL (s3://)
            storage_url = f"s3://{self.bucket_name}/{s3_key}"
            
            # Pre-signed URL 생성 (24시간 유효)
            download_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=86400  # 24시간
            )
            
            logger.info(f"Test script saved: {storage_url}")
            
            return {
                "filename": filename,
                "storage_url": storage_url,
                "download_url": download_url,
                "s3_key": s3_key
            }
        
        except Exception as e:
            logger.error(f"Failed to save test script: {e}")
            raise
    
    def get_test_script(
        self,
        s3_key: str
    ) -> str:
        """
        S3/MinIO에서 테스트 스크립트 다운로드
        
        Args:
            s3_key: S3 객체 키 (예: "my-app/login_20250109.spec.js")
        
        Returns:
            str: 테스트 코드 내용
        
        Example:
            >>> code = manager.get_test_script("my-app/login_20250109.spec.js")
            >>> print(code[:50])
            import { test, expect } from '@playwright/test'
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            content = response['Body'].read().decode('utf-8')
            logger.info(f"Test script retrieved: {s3_key}")
            return content
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"Test script not found: {s3_key}")
                raise FileNotFoundError(f"Test script not found: {s3_key}")
            else:
                logger.error(f"Failed to get test script: {e}")
                raise
    
    def delete_test_script(
        self,
        s3_key: str
    ) -> bool:
        """
        테스트 스크립트 삭제
        
        Args:
            s3_key: S3 객체 키
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"Test script deleted: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete test script: {e}")
            return False
    
    def list_test_scripts(
        self,
        repository_name: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        테스트 스크립트 목록 조회
        
        Args:
            repository_name: 저장소 이름 (필터링)
            limit: 최대 개수
        
        Returns:
            list: 파일 목록
                [
                    {
                        "key": str,
                        "size": int,
                        "last_modified": datetime
                    },
                    ...
                ]
        """
        try:
            # prefix 설정
            prefix = f"{repository_name}/" if repository_name else ""
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=limit
            )
            
            if 'Contents' not in response:
                return []
            
            files = [
                {
                    "key": obj['Key'],
                    "size": obj['Size'],
                    "last_modified": obj['LastModified']
                }
                for obj in response['Contents']
            ]
            
            logger.info(f"Listed {len(files)} test scripts")
            return files
        
        except Exception as e:
            logger.error(f"Failed to list test scripts: {e}")
            raise
    
    def get_download_url(
        self,
        s3_key: str,
        expires_in: int = 3600
    ) -> str:
        """
        다운로드용 Pre-signed URL 생성
        
        Args:
            s3_key: S3 객체 키
            expires_in: 유효 시간 (초)
        
        Returns:
            str: Pre-signed URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate download URL: {e}")
            raise


# 사용 예시
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # MinIO 예시
    print("\n=== Test 1: Initialize FileManager (MinIO) ===")
    try:
        manager = FileManager(
            endpoint_url="http://localhost:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            bucket_name="probe-tests"
        )
        print("✅ FileManager initialized")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("\nNote: Ensure MinIO is running:")
        print("  docker run -p 9000:9000 minio/minio server /data")
    
    # 테스트 코드 예시
    sample_code = """import { test, expect } from '@playwright/test';

test('login test', async ({ page }) => {
  await page.goto('https://example.com/login');
  await page.fill('#username', 'user');
  await page.fill('#password', 'pass');
  await page.click('button[type="submit"]');
  await expect(page).toHaveURL(/.*dashboard/);
});
"""
    
    print("\n=== Test 2: Save Test Script ===")
    try:
        result = manager.save_test_script(
            test_code=sample_code,
            repository_name="test-repo",
            test_name="login"
        )
        print(f"✅ Saved: {result['filename']}")
        print(f"   Storage URL: {result['storage_url']}")
        print(f"   Download URL: {result['download_url'][:50]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")